from numpy import array

def func1(a):
    a += 1

def func2(b):
    print(b)
    func1(b)
    print(b)

func2(array([1]))