import collections


# O(n) solution
def FreqNum(file):
    W, B = 0, 0
    L = []

    with open(file, 'r') as f:
        line = f.readline()
        l = line.split()
        W, B = int(l[0]), int(l[1])
        while True:
            line = f.readline()
            if not line:
                break
            else:
                pair = line.split()
                L.append((int(pair[0]), int(pair[1])))

    print("W = ", W, ", B = ", B)
    print("L = ", L)

    # in case the raw list is not sorted
    L.sort(key=lambda x: x[1])

    # window edge
    left, right = 0, 0

    # St
    res = []

    # determine the first window
    for i in range(len(L)):
        if L[0][1] + W > L[i][1]:
            res.append(L[i][0])
        else:
            right = i - 1
            break

    while True:
        # if found
        if max(collections.Counter(res).values()) >= B:
            print("S" + str(L[left][1]) + " = " + str(res))
            return True

        # if not found in the end
        if right + 1 == len(L):
            return False

        # move the window and find the next biggest window
        left += 1
        right += 1
        res.pop(0)
        res.append(L[right][0])
        while L[left][1] + W <= L[right][1]:
            left += 1
            res.pop(0)
        while right + 1 < len(L) and L[left][1] + W > L[right][1] and L[left][1] + W > L[right + 1][1]:
            right += 1
            res.append(L[right][0])


print("\nAnswer:", FreqNum('data/question.txt'))