def quickSort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    # 列表推导
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quickSort(left) + mid + quickSort(right)


print(quickSort([3, 6, 8, 10, 1, 2, 1]))
