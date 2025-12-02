import sys
import cv2
import numpy as np

from ids_peak import ids_peak, ids_peak_ipl_extension
from ids_peak_ipl import ids_peak_ipl


def configure_camera(nodemap):
    # ---- AcquisitionMode: Continuous ----
    try:
        acq_mode = nodemap.FindNode("AcquisitionMode")
        if acq_mode and acq_mode.AccessStatus() == ids_peak.NodeAccessStatus_ReadWrite:
            current_val = acq_mode.CurrentEntry().SymbolicValue()
            print("AcquisitionMode before:", current_val)
            # Find the "Continuous" entry
            entries = acq_mode.Entries()
            for entry in entries:
                if entry.SymbolicValue() == "Continuous":
                    acq_mode.SetCurrentEntry(entry)
                    break
            print("AcquisitionMode after:", acq_mode.CurrentEntry().SymbolicValue())
    except Exception as e:
        print("Could not set AcquisitionMode:", e)

    # ---- Exposure settings ----
    try:
        # Set auto exposure to Off for manual control
        exp_auto = nodemap.FindNode("ExposureAuto")
        if exp_auto and exp_auto.AccessStatus() == ids_peak.NodeAccessStatus_ReadWrite:
            entries = exp_auto.Entries()
            for entry in entries:
                if entry.SymbolicValue() == "Off":
                    exp_auto.SetCurrentEntry(entry)
                    print("ExposureAuto set to:", entry.SymbolicValue())
                    break
    except Exception as e:
        print("Could not set ExposureAuto:", e)
    
    # ---- Trigger: OFF (skip if not accessible) ----
    try:
        trig_mode = nodemap.FindNode("TriggerMode")
        if trig_mode and trig_mode.AccessStatus() == ids_peak.NodeAccessStatus_ReadWrite:
            print("TriggerMode before:", trig_mode.CurrentEntry().SymbolicValue())
            # Find "Off" entry
            entries = trig_mode.Entries()
            for entry in entries:
                if entry.SymbolicValue() == "Off":
                    trig_mode.SetCurrentEntry(entry)
                    print("TriggerMode after:", trig_mode.CurrentEntry().SymbolicValue())
                    break
        else:
            print("TriggerMode not writable, skipping")
    except Exception as e:
        print("Could not set trigger:", e)



def main():
    ids_peak.Library.Initialize()
    try:
        dm = ids_peak.DeviceManager.Instance()
        dm.Update()

        if len(dm.Devices()) == 0:
            raise RuntimeError("No devices found")

        dev = dm.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
        print("Opened:", dev.DisplayName())

        remote = dev.RemoteDevice().NodeMaps()[0]
        
        # Configure camera BEFORE starting acquisition
        configure_camera(remote)

        ds = dev.DataStreams()[0].OpenDataStream()

        payload_size = remote.FindNode("PayloadSize").Value()
        num_buffers = 4
        bufs = []

        for _ in range(num_buffers):
            b = ds.AllocAndAnnounceBuffer(payload_size)
            bufs.append(b)

        for b in bufs:
            ds.QueueBuffer(b)

        # Bring the host side up before locking TL params so it can tweak buffers
        ds.StartAcquisition()
        remote.FindNode("TLParamsLocked").SetValue(1)
        remote.FindNode("AcquisitionStart").Execute()

        # Get exposure time node for slider
        exposure_node = remote.FindNode("ExposureTime")
        if exposure_node:
            try:
                exp_min = int(exposure_node.Min())
                exp_max = int(exposure_node.Max())
                exp_current = int(exposure_node.Value())
                print(f"ExposureTime range: {exp_min} - {exp_max} us, current: {exp_current} us")
            except AttributeError:
                exp_min = 100
                exp_max = 100000
                exp_current = 10000
                print("ExposureTime min/max not available, using defaults")
        else:
            exp_min = 100
            exp_max = 100000
            exp_current = 10000
            print("ExposureTime node not found, using defaults")

        cv2.namedWindow("IDS live", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Exposure (us)", "IDS live", exp_current, exp_max, lambda x: None)
        cv2.createTrackbar("Threshold", "IDS live", 127, 255, lambda x: None)

        print("Press 'q' in the window to quit.")

        while True:
            try:
                buf = ds.WaitForFinishedBuffer(5000)  # 5 s timeout
            except ids_peak.TimeoutException as e:
                print("Timeout waiting for buffer:", e)
                # if this keeps happening, camera is not exposing; break to avoid infinite loop
                break

            # Set exposure from slider
            if exposure_node:
                exp_val = cv2.getTrackbarPos("Exposure (us)", "IDS live")
                exposure_node.SetValue(float(exp_val))

            # Wrap into IDS peak IPL image
            img = ids_peak_ipl.Image.CreateFromSizeAndBuffer(
                buf.PixelFormat(),
                buf.BasePtr(),
                buf.Size(),
                buf.Width(),
                buf.Height(),
            )

            # Convert to BGRa8, then to numpy
            img_bgra = img.ConvertTo(
                ids_peak_ipl.PixelFormatName_BGRa8,
                ids_peak_ipl.ConversionMode_Fast,
            )

            h, w = img_bgra.Height(), img_bgra.Width()
            np_img = img_bgra.get_numpy_1D().reshape(h, w, 4)
            np_bgr = np_img[:, :, :3]

            # Apply binary threshold
            thresh_val = cv2.getTrackbarPos("Threshold", "IDS live")
            gray = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

            cv2.imshow("IDS live", thresh)
            ds.QueueBuffer(buf)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        remote.FindNode("AcquisitionStop").Execute()
        ds.StopAcquisition()
        remote.FindNode("TLParamsLocked").SetValue(0)

        # Flush any remaining queued buffers before revoking
        ds.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
        
        for b in bufs:
            ds.RevokeBuffer(b)

        cv2.destroyAllWindows()

    finally:
        ids_peak.Library.Close()


if __name__ == "__main__":
    main()
