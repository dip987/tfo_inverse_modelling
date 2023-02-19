def yeet(func):
    def wrapper(*args):
        print("YEETED")
        func(*args)
        print("YEETED AGAIN")

    return wrapper


@yeet
def print_number(num):
    print("Hello", num)


if __name__ == '__main__':
    print_number(55)
