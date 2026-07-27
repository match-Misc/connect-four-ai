# Basic Skills

Additional information regarding the MuR620 can be found in the match-Wiki under "Versuchsaufbauten".

[⬆️ Back to Step-by-Step Guide](./step-by-step-guide.md)

## Changing the Battery of the MiR600

1. Stand in front of the flap on the left side of the MiR robot.
2. The flap is located at foot level. Press the two buttons on its left and right sides to open it.
3. The battery is located on the right side behind the opened flap.
4. Pull the small knob on the left to release the locking lever.
5. While holding the knob, push the lever (located to the right of the knob) downwards to disconnect the battery. This may require some force.
6. Once the connector is completely pushed down, you can release the knob on the left.
7. Pull the battery out by its handle. **Caution: The battery is long and very heavy!**
8. Connect the battery to the charger. The charger is usually located on the table next to the Stäubli robot.

[⬆️ Back to Step-by-Step Guide](./step-by-step-guide.md)

## Manually Moving the MuR620

1. Locate the three buttons on the back left of the MiR600 robot and press the **Start button** (the leftmost one).
INSERT IMAGE OF THE THREE BUTTONS ON THE BACK LEFT OF THE MIR600
2. Try pushing the MuR620 to see if it moves freely.
  - **If it does not move:**
    3. Find the two buttons for opening the rear latch on the back of the MiR600. Press both buttons to open the latch.
    INSERT IMAGE
    4. Turn the switch located behind the latch, then try pushing the robot again. It should now move easily.
    INSERT IMAGE
  - **Once it moves freely:**
    5. Push the robot to your desired destination.
    6. Wait until the MiR600 is completely booted (indicated by a solid red light due to the active emergency stop). Then, press and hold the **Start button** until the lights indicate that the MiR600 is shutting down.

[⬆️ Back to Step-by-Step Guide](./step-by-step-guide.md)

## How to Turn On the MuR620

1. **(Optional but recommended for longer sessions):** Find a standard C13 power cable and plug it into the front of the MiR620. This charges the additional top-hat battery, not the internal MiR600 battery.
2. Open the rear right door of the MuR620 top module. Inside, locate the control cabinet and ensure its main lever is turned to **"1" (ON)**.
3. Check the back control panel of the MuR620 and verify that the key switch is turned to the **"ON"** position.
   > [INSERT IMAGE SHOWING THE BACK CONTROL PANEL OF THE MIR600 WHERE THE KEY SWITCH IS LOCATED]

[⬆️ Back to Step-by-Step Guide](./step-by-step-guide.md)

## How to Turn On the UR10

1. Ensure the MuR620 is turned on (see the previous section).
2. Press the button labeled **"UR10L"** (left) or **"UR10R"** (right), depending on which robot arm you need to power on.

[⬆️ Back to Step-by-Step Guide](./step-by-step-guide.md)

## Accessing the UR User Interface

1. Connect a portable touch monitor to the back of the MiR620 using both USB and HDMI cables.
   > [INSERT IMAGE OF THE TOUCH MONITOR CONNECTED TO THE MIR620]
2. Check the display for a four-way split screen and identify which quadrant corresponds to your UR interface.
3. Open the rear left door of the MuR620 top module.
4. Locate the KVM switch inside and press its button repeatedly until your specific UR interface is displayed in full screen.
   > [INSERT IMAGE OF KVM LOCATION AND THE BUTTON]

[⬆️ Back to Step-by-Step Guide](./step-by-step-guide.md)

## Unlocking the UR Robot

1. Make sure you have started the MuR620 robot. See: [How to Turn On the MuR620](#how-to-turn-on-the-mur620)
2. Locate the toggle switch on the back of the MuR620 (used for putting the mobile robot into local emergency stop mode) and ensure it is in the correct operational state: Local (not Remote).
3. Verify that the emergency stop (e-stop) button on the back of the MiR is **not** activated (release it if necessary). And if the UR Teach Pendant is connected, its red emergency stop button (at the top of the pendant) must be released.
4. On the UR touch screen, a window might be shown saying "Robot Emergency Stop". If so, click the "Go to initialization screen". If no, press the button on the bottom left with the big red circle saying "Robot Emergency Stop".
5. If all emergency stops are released, you should be able to press the "On" button which will startup the robot.
6. Once the robot has fully booted, press **"START"**. You should hear a distinct click (a "knick-knack" sound) indicating that the brakes are released and the joints are now unlocked.
7. Press "Exit" to exit the initialization window.

[⬆️ Back to Step-by-Step Guide](./step-by-step-guide.md)

## Moving the UR Robot

1. Make sure you have unlocked the UR robot. See: [Unlocking the UR Robot](#unlocking-the-ur-robot)
2. On the UR touch screen, tap the **"Move"** tab located in the top left corner.
3. You can now jog the robot manually. Use the controls on the right side of the screen to move the robot either in **Joint Space** (controlling individual joints) or **Cartesian Space** (moving the tool center point linearly).
   > [INSERT SCREENSHOT OF THE MOVE TAB]

[⬆️ Back to Step-by-Step Guide](./step-by-step-guide.md)

## Load UR Installation and UR program 

   > [MARK THIS TO BE CONFIRMED LATER!]

1. On the UR touch screen, tap the **Open** button (a folder icon in the top header).
2. Select the correct **Connect Four program file** (`.urp`) from the file manager and tap **Open**.
3. If prompted to load the associated installation file (`.installation`), tap **Yes** to confirm.
4. Once loaded, press the **Play** (▶️) button at the bottom of the screen to start the program.

[⬆️ Back to Step-by-Step Guide](./step-by-step-guide.md)
