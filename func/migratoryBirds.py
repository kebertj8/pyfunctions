# arr = [1,1,2,2,3]
# two types of 1 
# two types of 2 
# one sighting of type 3 
# pick the lower of the two types seen twice which is type #1 
# int arr[n]: the types of birds sighted
# constraints: 5<=n<=2*10^5
# It is guaranteed that each type is 1,2,3,4, or 5 

# def mostFrequent(arr, n):

#     # Sort the array
#     arr.sort()

#     # find the max frequency using
#     # linear traversal
#     max_count = 1
#     res = arr[0]
#     curr_count = 1

#     for i in range(1, n):
#         if (arr[i] == arr[i - 1]):
#             curr_count += 1

#         else:
#             if (curr_count > max_count):
#                 max_count = curr_count
#                 res = arr[i - 1]

#             curr_count = 1

#     # If last element is most frequent
#     if (curr_count > max_count):

#         max_count = curr_count
#         res = arr[n - 1]

#     return res


# # Driver Code
# arr = [1, 5, 2, 1, 3, 2, 1]
# n = len(arr)
# print(mostFrequent(arr, n))

# input()
# count = [0]*6
# for t in map(int, input().strip().split()):
#     count[t] += 1
# print(count.index(max(count)))

# import sys
# from collections import Counter

# n = int(input().strip())
# types = list(map(int, input().strip().split(' ')))

# birds = Counter(types)  # Counts the array into a dictionary
# print(birds.most_common(1)[0][0])
