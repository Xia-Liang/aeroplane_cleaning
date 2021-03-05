try:
    from config import *
except ImportError:
    raise RuntimeError('cannot import config file')


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', FPS)
        self._queues = []
        self._settings = None

    def __enter__(self):
        # some data about the simulation such as synchrony between client and server or rendering mode
        self._settings = self.world.get_settings()
        # ---- This is important carla.WorldSettings
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        # This method only has effect on synchronous mode, when both client and server move together.
        # The client tells the server when to step to the next frame and returns the id of the newly started frame.
        self.frame = self.world.tick()
        # get the data synchronous data
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            # client timestamp == world timestamp
            if data.frame == self.frame:
                return data
