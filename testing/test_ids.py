# import cv2
# import numpy as np
from ids_peak import ids_peak

def main():

    # Initialize IDS peak library
    ids_peak.Library.Initialize()

    # Create and update the device manager
    device_manager = ids_peak.DeviceManager.Instance()
    device_manager.Update()

    if device_manager.Devices().empty():
        raise RuntimeError("No IDS camera found!")

    # Open first available camera
    device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
    print("Connected to:", device.ModelName(), " SN:", device.SerialNumber())

    # Get the camera's data stream
    datastream = device.DataStreams()[0].OpenDataStream()

    # Prepare buffers
    payload_size = device.RemoteDevice().NodeMap().FindNode("PayloadSize").Value()
    buffers = []
    for _ in range(4):
        buf = datastream.AllocAndAnnounceBuffer(payload_size)
        buffers.append(buf)
        datastream.QueueBuffer(buf)

    # Start acquisition
    device.RemoteDevice().NodeMap().FindNode("AcquisitionStart").Execute()
    datastream.StartAcquisition()
    print("Streaming... press CTRL+C to stop.")

    try:
        while True:
            buffer = datastream.WaitForFinishedBuffer(1000)

            # Convert IDS buffer → NumPy array
            np_image = np.ctypeslib.as_array(buffer.GetBuffer().ctypes.data_as(
                np.ctypeslib.ctypes.POINTER(np.uint8)),
                shape=(buffer.Height(), buffer.Width(), 1))

            frame = cv2.cvtColor(np_image, cv2.COLOR_BAYER_RG2BGR)

            cv2.imshow("IDS Stream", frame)
            if cv2.waitKey(1) == 27:  # ESC
                break

            datastream.QueueBuffer(buffer)

    finally:
        # Cleanup
        datastream.StopAcquisition()
        device.RemoteDevice().NodeMap().FindNode("AcquisitionStop").Execute()

        for buf in buffers:
            datastream.RevokeBuffer(buf)

        device.Close()
        ids_peak.Library.Close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
