class NOTE(object):
    def __init__(self, start, end, pitch, velocity, Type):
        self.velocity = velocity
        self.pitch = pitch
        self.start = start
        self.end = end
        self.Type = Type
    def get_duration(self):
        return self.end - self.start
    def __repr__(self):
        return f'Note(start={self.start}, end={self.end}, pitch={self.pitch}, velocity={self.velocity}, Type={self.Type})'
