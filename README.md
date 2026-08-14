# Schlag den Roboter - Vier gewinnt / Beat the Robot - Connect Four

This repository contains a robot-based version of the classic game **Connect Four**, allowing a human to play directly against a robotic opponent. 

It builds upon the open-source project [connect-four-ai by benjaminrall](https://github.com/benjaminrall/connect-four-ai) and extends it with custom code for the **UR10e robot arm**. While originally designed as a showcase for events like the "Nacht des Maschinenbaus" (Night of Mechanical Engineering) or "Die Nacht die WissenSchaft" (The Night of Science), the setup is perfectly suited for various other exhibitions.

<img src="./docs/images/game-nobackground.png" alt="Connect Four game on the MuR620d" width="400" />

This repository and the documentation below provide all the necessary components and instructions to set up the game quickly and smoothly.

## Step-by-Step Setup Guide

Follow these steps in order to set up the Connect Four game:

### 1. Preparations

Before starting the setup, the following items should be prepared:
- **MuR620d** (check the naming on the back of the mobile robot)
- **Grey Box** (labeled "Schlag den Roboter - Vier Gewinnt" in one of the cabinets in Scale)
- **Schunk Co-act Gripper** (See [Mount Schunk Gripper to UR10 1](./docs/hardware-setup.md#mount-schunk-gripper-to-ur10-1) for image)

### 2. Hardware Assembly

- [Positioning the MuR620](./docs/hardware-setup.md#positioning-the-mur620)
- [Mounting of hardware components](./docs/hardware-setup.md#mounting-of-hardware-components)
- [Cabling and Power Supply](./docs/hardware-setup.md#cabling-and-power-supply)

### 3. Software & Calibration

- [Getting the NUC ready](./docs/hardware-setup.md#getting-the-nuc-ready)
- [Calibrate the Camera for the Game](./docs/hardware-setup.md#calibrate-the-camera-for-the-game)

### 4. Robot Preparation

- [Getting the UR-robots ready](./docs/hardware-setup.md#getting-the-ur-robots-ready)
  - [Move UR10 2 to the Side](./docs/hardware-setup.md#move-ur10-2-to-the-side)
  - [Mount Schunk Gripper to UR10 1](./docs/hardware-setup.md#mount-schunk-gripper-to-ur10-1)

### 5. Launch

- [Start UR program and move to initial pose](./docs/hardware-setup.md#start-ur-program-and-move-to-initial-pose)
- [Start the Game](./docs/hardware-setup.md#start-the-game)


## Instructor Manual (Betreuer-Handbuch)

As an instructor, your responsibilities during the game are:

- Tell the person who wants to play how to set the difficulty by pressing the button.
- Instruct players on using the NFC reader: By placing their NFC chip on the reader, they can set the difficulty and then start the game to get ranked on the leaderboard.
- Explain how to put in the chips.
- **Instruct the player to wait for a second after the robot places its chip** before making their next move.
- **Remind the player to watch out for robot movements.** Even though it is a collaborative robot (cobot) and generally safe, we don't want to take any unnecessary risks.
- When the game is won or lost, pull out the aluminum bar at the bottom of the game to eject all stones. This will automatically reset the game.
- Sort the **green** chips into the chip tower for the player.
- Sort the **black** chips into the chip tray of the robot.
- ⚠️ **Safety First:** Only after everything has been reset is the next person allowed to start. Otherwise, the robot might move and pose a safety risk for the person sorting the chips.
- If you want to reset the game early (before the game is finished), just pull out the aluminum bar at the bottom to eject all stones. This will automatically reset the game so the human can start again.

## Additional Information and Bug Fixing

If you run into issues or need to perform basic operations, refer to the following guides:

- **[Basic Skills Guide](./docs/basic-skills.md)**: Covers manual movement, turning the robots on/off, and using the UR user interface.
- **[Understanding the UR Robot Program](./docs/robot-program.md)**: Explains the robot program. Covers teaching positions, connection to the NUC, DIOs.
- **[FAQ](./docs/faq.md)**: Common questions and troubleshooting.


## Documentation

Comprehensive documentation is available in the [`docs/`](./docs/) folder:

- **[Connect Four AI Documentation](./docs/connect-four-ai.md)** - Complete technical documentation of the AI algorithm, performance benchmarks, and implementation details
- **[Software Setup Guide](./docs/software-setup.md)** - Step-by-step instructions for setting up the development environment and installing dependencies
- **[Hardware Setup Guide](./docs/hardware-setup.md)** - Physical setup instructions for robotic implementations and hardware requirements
- **[Basic Skills](./docs/basic-skills.md)** - Basic operations for the robotic system
- **[Understanding the UR Robot Program](./docs/robot-program.md)** - How the UR program works, teaching the tray and board positions, socket connection to the NUC, and the digital I/Os


