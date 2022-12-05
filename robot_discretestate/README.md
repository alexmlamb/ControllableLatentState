# Robot control in latent space
1. Collect data to learn the controllable latent state
   - The robotic arm's end effector was setup to move inside a 9 cell grid in front of the robot.  The joint positions for centroids of the 9 cell were recorded.  Five high level commands were used to navigate the grid.
     -  "Move North"
     -  "Move South"
     -  "Move East"
     -  "Move West"
     -  "Stay in place" 
   - A random policy was used to explore the grid.  Each time step consisted of 
     - Capturing a before image
     - Executing the move command on the robot
     - Capturing an after image
     - Recording the data to a csv file.
   - Experimental data is available here: [Data Set](https://microsoft-my.sharepoint.com/:u:/p/ranaras/EY3HbwFmY7BMrERzub-7MfwB350F99oxtRX1SL9c1z1MlA?e=PdycZ0)
2. Train a model to extract controllable latent state
3. Creating Latent state graph
4. Using image to get a latent state (LS)
5. Planning - Find a path from LS1 to LS2
6. Executing a plan on the robot 