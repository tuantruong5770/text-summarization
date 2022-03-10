import time


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        out = func(*args, **kwargs)
        sec = time.time() - start
        total_time = beautify_time(sec)
        print(f'***** Total time elapsed: - {total_time} *****')
        return out
    return wrapper


def beautify_time(seconds):
    sec = seconds
    hour = sec // 3600
    sec = sec - (3600 * hour)
    min = sec // 60
    sec = sec - (60 * min)
    return f'{int(hour)}:{int(min)}:{sec}'


# def log(string)

if __name__ == '__main__':
    pass
