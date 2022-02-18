# Bidirectional_RRT
Planning a path in a 3D space with 6 degree of freedom using bidirectional RRT

# Environment
The environment consists of a flat square field, 250 meters on a side, filled with obstacles. The obstacles consist of large patches of un-navigable thick brush, trees, and weeds, suspiciously shaped like giant tetrominoes. While the environment is not specifically a grid, the base dimension for each obstacle square unit is 15 meters. Inside this field, a firetruck operates, attempting to extinguish fires that emerge.
Starting at time 0 and at 60 second intervals, an arsonist/wumpus sets a major conflagration at a random obstacle. This sets the obstacle state to burning. After 20 seconds in this state, the obstacle sets all obstacles within a 30 meter radius to the state of burning. Run the simulation for 3600 seconds.
The truck starts at a random point in the map. If the truck stops within 10 meters of a burning obstacle, it sets the state to extinguished. Use a path planner to drive the truck to desired locations and attempt to extinguish as many obstacles as possible.

