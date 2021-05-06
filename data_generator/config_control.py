"""
CARLA manual control.

Use WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

try:
    from config import *
except ImportError:
    raise ImportError('cannot import config file')


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================

class KeyboardControl(object):
    """Class that handles keyboard input."""

    def __init__(self, player):

        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        self.player = player

    def parse_events(self, clock):
        self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
        self.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        # move forward or backward
        if keys[K_w]:
            self._control.reverse = False
            self._control.throttle = min(self._control.throttle + 0.2, 0.6)
        elif keys[K_q]:
            self._control.reverse = True
            self._control.throttle = min(self._control.throttle + 0.2, 0.6)
        else:
            self._control.throttle = 0.0

        # brake
        if keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        # turn left or right
        steer_increment = 1e-3 * milliseconds
        if keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 7)


def should_quit():
    """
    stop event
    :return:
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False
