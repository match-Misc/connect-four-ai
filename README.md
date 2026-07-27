# Schlag den Roboter - Vier gewinnt / Beat the Robot - Connect Four

This repository contains a robot-based version of the classic game **Connect Four**, allowing a human to play directly against a robotic opponent. 

It builds upon the open-source project [connect-four-ai by benjaminrall](https://github.com/benjaminrall/connect-four-ai) and extends it with custom code for the **UR10e robot arm**. While originally designed as a showcase for events like the "Nacht des Maschinenbaus" (Night of Mechanical Engineering) or "Die Nacht die WissenSchaft" (The Night of Science), the setup is perfectly suited for various other exhibitions.

> [INSERT IMAGE OF THE GAME ON THE MIR ROBOT]

This repository and the documentation below provide all the necessary components and instructions to set up the game quickly and smoothly.

## Step-by-Step Setup Guide

Follow these steps in order to set up the Connect Four game:

1. [Preparations](#preparations)
2. [Positioning the MuR620](#positioning-the-mur620)
3. [Mounting of hardware components](#mounting-of-hardware-components)
4. [Getting the UR-robots ready](#getting-the-ur-robots-ready)
   - [Move UR10 2 to the Side](#move-ur10-2-to-the-side)
   - [Mount Schunk Gripper to UR10 1](#mount-schunk-gripper-to-ur10-1)
   - [Start UR program and move to initial pose](#start-ur-program-and-move-to-initial-pose)
5. [Basic Knowledge](#basic-knowledge)
   - [Changing the Battery of the MiR600](#changing-the-battery-of-the-mir600)
   - [Manually Moving the MuR620](#manually-moving-the-mur620)
   - [How to Turn On the MuR620](#how-to-turn-on-the-mur620)
   - [How to Turn On the UR10](#how-to-turn-on-the-ur10)
   - [Accessing the UR User Interface](#accessing-the-ur-user-interface)
   - [Unlocking the UR Robot](#unlocking-the-ur-robot)
   - [Moving the UR Robot](#moving-the-ur-robot)
   - [Load UR Installation and UR program](#load-ur-installation-and-ur-program)
6. [Documentation](#-documentation)
7. [Quick Start](#-quick-start)
8. [Current start](#current-start)


[⬆️ Back to Step-by-Step Guide](#step-by-step-setup-guide)

## Preparations
Before starting the setup, the following items should be prepared:
- MiR600d (see the naming on the back of the mobile robot)
- Grey Box with the "Schlag den Roboter - Vier Gewinnt" description.
- Schunk Gripper? (check for the naming and brand)

[⬆️ Back to Step-by-Step Guide](#step-by-step-setup-guide)

## Positioning the MuR620

The first step is to bring the MuR620 to its designated operating location. This is done by manually moving the robot. For detailed instructions, see: [Manually Moving the MuR620](#manually-moving-the-mur620).

**Important Notes Regarding the Location:**
- Avoid placing the setup in front of very bright or overexposed backgrounds, as this may interfere with the camera's image recognition.

[⬆️ Back to Step-by-Step Guide](#step-by-step-setup-guide)

## Mounting of hardware components
To ensure the robot program works without modifications, all hardware components **must be mounted at the exact same screw holes**. 
The image below shows a top view of the MuR620 and its mounting grid plate. Please pay attention to the orientation: **the right side of the image is the front** of the robot. 

<img src="./docs/images/connect-four-mounting-layout-top-view.png" alt="Connect Four Mounting Layout Top View" width="600" />

Ideally, you should start counting the holes on the grid plate from the bottom right edge to determine exactly which hole each component should be screwed into. For example, the front screw of the Connect Four board must be fastened in the first row, fourth column from the right.
The components to be mounted on the plate are color-coded as follows:
- 🟢 **Green (Connect Four Board)**: The surface with the black sticker must face the RealSense camera.
- 🟣 **Purple (Chip Collection Tray)**: Prevents chips from falling onto the floor when the game board's gate is opened.
- 🟠 **Orange (Black Chip Holder)**: The black chips are placed here in an orderly fashion so the robot can pick them up.
- 🔴 **Red (RealSense Camera Mount)**: The slanted sides must face away from the game board.
- 🔵 **Blue (UR Robot Arm)**: The Universal Robot (UR) arm that is used to play the game.

INSERT REAL IMAGE OF THE GAME ON THE MIR ROBOT


[⬆️ Back to Step-by-Step Guide](#step-by-step-setup-guide)


## Getting the UR-robots ready
Only one of the MuR620's robot arms is needed for the game; the other arm must be moved out of the way. In our setup, the UR10 on the left side of the MuR620 is designated as **UR10 1**, and the one on the right side is **UR10 2**.

- **UR10 2** is not needed and must be parked safely to the side.
- **UR10 1** is responsible for playing the game and will be equipped with the appropriate gripper and jaws.

Detailed instructions are provided in the following steps:

1. [Move UR10 2 to the Side](#move-ur10-2-to-the-side)
2. [Mount Schunk Gripper to UR10 1](#mount-schunk-gripper-to-ur10-1)
3. [Start UR program and move to initial pose](#start-ur-program-and-move-to-initial-pose)

[⬆️ Back to Step-by-Step Guide](#step-by-step-setup-guide)

### Move UR10 2 to the Side

The **UR10 2** [INSERT IMAGE LINK HERE] is not used in the Connect Four game and must be moved out of the way so it doesn't interfere with **UR10 1**.

**Prerequisites:**
1. Make sure the MuR620 is turned on. See: [How to Turn On the MuR620](#how-to-turn-on-the-mur620)
2. Make sure the UR is unlocked. See: [Unlocking the UR Robot](#unlocking-the-ur-robot)
3. Familiarize yourself with moving the UR manually. See: [Moving the UR Robot](#moving-the-ur-robot)

**Steps to move UR10 2:**
1. Access the UR User Interface by switching through the KVM switch and ensure you are controlling **UR10 2**.
2. Use the **Move** tab to jog UR10 2 safely to the side so that its workspace does not overlap with UR10 1.
3. See reference image for a pose that does not interfere with the game
INSERT IMAGE HERE


[⬆️ Back to Step-by-Step Guide](#step-by-step-setup-guide)

### Mount Schunk Gripper to UR10 1
1. Look in the grey box for the two gripper jaws.
INSERT IMAGE OF THE GRIPPER JAWS
2. Mount the two gripper jaws to the Schunk gripper.
3. The gripper has an aluminium plate on the back with four diagonal holes. REMOVE IT from the gripper by loosening the four screws on the back. 
INSERT IMAGE SHOWING THE HOLES
4. Mount the aluminium plate directly to the UR10 1 (see the image below).
5. Mount the Schunk gripper to the aluminium plate. Using the diagonal holes going through the Schunk gripper 


[⬆️ Back to Step-by-Step Guide](#step-by-step-setup-guide)

### Start UR program and move to initial pose
**Important:** Before proceeding, verify that the correct program and installation are loaded. See: [Load UR Installation and UR program](#load-ur-installation-and-ur-program).

1. On the UR touch screen, tap the **"Program"** button in the top left corner.
2. For safety, reduce the robot's speed by dragging the speed slider (at the bottom of the screen) down to approximately **20%**.
3. Press the **"Play"** button located in the bottom right corner, then select **"Play from beginning: Robot Program"**.
4. You will be prompted to move the robot to its starting position. 
   - **Caution:** Make sure the robot's path is completely clear of obstacles.
   - Press and hold the **"Move robot to: ..."** button. The robot will move *only* while you are holding the button.
5. Once the robot reaches the start position, a prompt will appear. Tap **"Play from: Robot Program"** to fully start the program. 
   - *Note:* The robot may immediately pick up a stone. This is normal behavior indicating it is successfully connected to the NUC and waiting for commands.

The robot is now actively waiting for TCP commands from the NUC to determine where to place the stones.
**The UR setup is now finished.**

[⬆️ Back to Step-by-Step Guide](#step-by-step-setup-guide)


## Basic Knowledge

Additional information regarding the MuR620 can be found in the match-Wiki under "Versuchsaufbauten".


[⬆️ Back to Step-by-Step Guide](#step-by-step-setup-guide)


### Changing the Battery of the MiR600

1. Stand in front of the flap on the left side of the MiR robot.
2. The flap is located at foot level. Press the two buttons on its left and right sides to open it.
3. The battery is located on the right side behind the opened flap.
4. Pull the small knob on the left to release the locking lever.
5. While holding the knob, push the lever (located to the right of the knob) downwards to disconnect the battery. This may require some force.
6. Once the connector is completely pushed down, you can release the knob on the left.
7. Pull the battery out by its handle. **Caution: The battery is long and very heavy!**
8. Connect the battery to the charger. The charger is usually located on the table next to the Stäubli robot.


[⬆️ Back to Step-by-Step Guide](#step-by-step-setup-guide)


### Manually Moving the MuR620

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


[⬆️ Back to Step-by-Step Guide](#step-by-step-setup-guide)


### How to Turn On the MuR620

1. **(Optional but recommended for longer sessions):** Find a standard C13 power cable and plug it into the front of the MiR620. This charges the additional top-hat battery, not the internal MiR600 battery.
2. Open the rear right door of the MuR620 top module. Inside, locate the control cabinet and ensure its main lever is turned to **"1" (ON)**.
3. Check the back control panel of the MuR620 and verify that the key switch is turned to the **"ON"** position.
   > [INSERT IMAGE SHOWING THE BACK CONTROL PANEL OF THE MIR600 WHERE THE KEY SWITCH IS LOCATED]


[⬆️ Back to Step-by-Step Guide](#step-by-step-setup-guide)


### How to Turn On the UR10

1. Ensure the MuR620 is turned on (see the previous section).
2. Press the button labeled **"UR10L"** (left) or **"UR10R"** (right), depending on which robot arm you need to power on.


[⬆️ Back to Step-by-Step Guide](#step-by-step-setup-guide)


### Accessing the UR User Interface

1. Connect a portable touch monitor to the back of the MiR620 using both USB and HDMI cables.
   > [INSERT IMAGE OF THE TOUCH MONITOR CONNECTED TO THE MIR620]
2. Check the display for a four-way split screen and identify which quadrant corresponds to your UR interface.
3. Open the rear left door of the MuR620 top module.
4. Locate the KVM switch inside and press its button repeatedly until your specific UR interface is displayed in full screen.
   > [INSERT IMAGE OF KVM LOCATION AND THE BUTTON]


[⬆️ Back to Step-by-Step Guide](#step-by-step-setup-guide)


### Unlocking the UR Robot

1. Make sure you have started the MuR620 robot. See: [How to Turn On the MuR620](#how-to-turn-on-the-mur620)
2. Locate the toggle switch on the back of the MuR620 (used for putting the mobile robot into local emergency stop mode) and ensure it is in the correct operational state: Local (not Remote).
3. Verify that the emergency stop (e-stop) button on the back of the MiR is **not** activated (release it if necessary). And if the UR Teach Pendant is connected, its red emergency stop button (at the top of the pendant) must be released.
4. On the UR touch screen, a window might be shown saying "Robot Emergency Stop". If so, click the "Go to initialization screen". If no, press the button on the bottom left with the big red circle saying "Robot Emergency Stop".
5. If all emergency stops are released, you should be able to press the "On" button which will startup the robot.
6. Once the robot has fully booted, press **"START"**. You should hear a distinct click (a "knick-knack" sound) indicating that the brakes are released and the joints are now unlocked.
7. Press "Exit" to exit the initialization window.

[⬆️ Back to Step-by-Step Guide](#step-by-step-setup-guide)

### Moving the UR Robot

1. Make sure you have unlocked the UR robot. See: [Unlocking the UR Robot](#unlocking-the-ur-robot)
2. On the UR touch screen, tap the **"Move"** tab located in the top left corner.
3. You can now jog the robot manually. Use the controls on the right side of the screen to move the robot either in **Joint Space** (controlling individual joints) or **Cartesian Space** (moving the tool center point linearly).
   > [INSERT SCREENSHOT OF THE MOVE TAB]

[⬆️ Back to Step-by-Step Guide](#step-by-step-setup-guide)

### Load UR Installation and UR program 

   > [MARK THIS TO BE CONFIRMED LATER!]

1. On the UR touch screen, tap the **Open** button (a folder icon in the top header).
2. Select the correct **Connect Four program file** (`.urp`) from the file manager and tap **Open**.
3. If prompted to load the associated installation file (`.installation`), tap **Yes** to confirm.
4. Once loaded, press the **Play** (▶️) button at the bottom of the screen to start the program.

[⬆️ Back to Step-by-Step Guide](#step-by-step-setup-guide)


### 📚 Documentation

Comprehensive documentation is available in the [`documentation/`](./documentation/) folder:

- **[Connect Four AI Documentation](./documentation/connect-four-ai.md)** - Complete technical documentation of the AI algorithm, performance benchmarks, and implementation details
- **[Software Setup Guide](./documentation/software-setup.md)** - Step-by-step instructions for setting up the development environment and installing dependencies
- **[Hardware Setup Guide](./documentation/hardware-setup.md)** - Physical setup instructions for robotic implementations and hardware requirements


[⬆️ Back to Step-by-Step Guide](#step-by-step-setup-guide)


## 🚀 Quick Start

ToDo


[⬆️ Back to Step-by-Step Guide](#step-by-step-setup-guide)


## Current start

python3 web-ui/app/app.py

Wait 10 seconds

python detection.py

Next: add rgb max values to the calibration and save it in the config file.
[⬆️ Back to Step-by-Step Guide](#step-by-step-setup-guide)


