class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


class CellEntity(Enum):
    # entity encodings for grid observations
    OUT_OF_BOUNDS = 0
    EMPTY = 1
    FOOD = 2
    AGENT = 3

    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "level", "history", "reward", "is_self"]
    ) 

// grid_shape_x, grid_shape_y = self.field_size -> field_size is the size of the map.
// sight -> how far can the agent see.
// agents_layer, foods_layer, access_layer
// self.field -> contains the whole matrix.

'''
        # I dont want to spwan a food which the bottom 3 players combined can't lift.
        self.spawn_food(
            self.max_food, max_level=sum(player_levels[:3]) 
        )
'''
RUN ->
python3.10 py_run_env.py