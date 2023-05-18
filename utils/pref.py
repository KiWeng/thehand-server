from time import perf_counter, sleep


class DotTimer:
    def __init__(self):
        self.count = 0
        self.avg = 0
        self.prev = None

    def dot(self):
        if self.prev is None:
            self.prev = perf_counter()
        else:
            current = perf_counter()
            delta = current - self.prev
            self.prev = current
            self.count += 1
            self.avg = (self.avg * (self.count - 1) + delta) / self.count


if __name__ == "__main__":
    dt = DotTimer()
    for i in range(100):
        sleep(1)
        dt.dot()
        print(dt.avg)
