# import heapq

# class node:
#     def __init__(self, freq, symbol, left = None, right = None) :
#         self.freq = freq
#         self.symbol = symbol
#         self.left = left
#         self.right = right
#         self.huff = ''
        
#     def __lt__(self, nxt):
#         return self.freq < nxt.freq
    
    

# def printNodes(node, val=''):
#     newval = val + str(node.huff)
    
#     if(node.left):
#         printNodes(node.left, newval)
#     if(node.right):
#         printNodes(node.right, newval)
#     if(not node.left and not node.right):
#         print(f'{node.symbol} --> {newval}')
    
    
# chars = ['a', 'b', 'c', 'd', 'e', 'f']
# freq = [ 5, 9, 12, 13, 16, 45]
# nodesList = []


# for i in range(len(chars)):
#     heapq.heappush(nodesList, node(freq[i], chars[i]))
    

# while len(nodesList) > 1:
#     left = heapq.heappop(nodesList)
#     right = heapq.heappop(nodesList)
    
#     left.huff = 0
#     right.huff = 1

#     newnode = node(left.freq+right.freq, left.symbol+right.symbol, left, right)
    
#     heapq.heappush(nodesList, newnode)
    

# printNodes(nodesList[0])


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''##################################'''
# knapsack

# class Item:
    
#     def __init__(self, value, weight):
#         self.value = value
#         self.weight = weight
        

# def fractknapsack(w, arr):
    
#     arr.sort(key = lambda x:(x.value/x.weight), reverse= True)
    
#     finalval = 0.0
    
#     for item in arr:
#         if item.weight <= w:
#             w -= item.weight
#             finalval += item.value
#         else:
#             finalval += item.value * (w/item.weight)
#             break
        
#     return finalval

# w = 20

# arr = [Item(25, 18), Item(24, 15), Item(15, 10)]


# print(fractknapsack(w, arr))
            
            
            
# nqueen
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''##################################'''


# def isSafe(mat, r,c):
#     for i in range(r):
#         if mat[i][c] == 'Q':
#             return False
        
#     (i,j) = (r,c)
#     while i>=0 and j>=0:
#         if mat[i][j] == 'Q':
#             return False
        
#         i -= 1
#         j -= 1
        
#     i,j = r,c
#     while i>=0 and j<len(mat):
#         if mat[i][j] == 'Q':
#             return False
        
#         i -= 1
#         j += 1
        
#     return True


# def printSol(mat):
#     for i in mat:
#         print(str(i))
#     print()


# def nqueen(mat, r, final):
#     if r == len(mat):
#         printSol(mat)
#         final[0] = True
#         return
    
#     for c in range(len(mat)):
#         if not final[0] and isSafe(mat, r, c):
#             mat[r][c] = 'Q'
#             nqueen(mat, r+1, final)
#             mat[r][c] = '-'
            

# n = int(input('enter:'))

# mat = [['-' for x in range(n)]for y in range(n)]

# final = [False]
# nqueen(mat,0,final)

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''##################################'''

            