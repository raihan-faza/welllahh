def rotateLeft(d, arr):
    new_arr = []
    for x in range(-d, len(arr) - d):
        new_arr.append(arr[x])
    return new_arr


print(
    rotateLeft(
        10,
        [41, 73, 89, 7, 10, 1, 59, 58, 84, 77, 77, 97, 58, 1, 86, 58, 26, 10, 86, 51],
    )
)
