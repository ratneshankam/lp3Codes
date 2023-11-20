def isSafe(mat, r, c):
    for i in range(r):
        if mat[i][c] == 'Q':
            return False
    (i, j) = (r, c)
    while i >= 0 and j >= 0:
        if mat[i][j] == 'Q':
            return False
        i = i - 1
        j = j - 1
    (i, j) = (r, c)
    while i >= 0 and j < len(mat):
        if mat[i][j] == 'Q':
            return False
        i = i - 1
        j = j + 1
    return True
 
def printSolution(mat):
    for r in mat:
        print(str(r).replace(',', '').replace('\'', ''))
    print()
  

def nQueen(mat, r):
    if r == len(mat):
        printSolution(mat)
        return
    for i in range(len(mat)):
        if isSafe(mat, r, i):
            mat[r][i] = 'Q'
            nQueen(mat, r + 1)
            mat[r][i] = '-'
    
if __name__ == '__main__':
    N = int(input())
    mat = [['-' for x in range(N)] for y in range(N)]
    nQueen(mat, 0)
 
 
# def isSafe(mat, r, c):
#     for i in range(r):
#         if mat[i][c] == 'Q':
#             return False
#     (i, j) = (r, c)
#     while i >= 0 and j >= 0:
#         if mat[i][j] == 'Q':
#             return False
#         i = i - 1
#         j = j - 1
#     (i, j) = (r, c)
#     while i >= 0 and j < len(mat):
#         if mat[i][j] == 'Q':
#             return False
#         i = i - 1
#         j = j + 1
#     return True

# def printSolution(mat):
#     for r in mat:
#         print(str(r).replace(',', '').replace('\'', ''))
#     print()

# def nQueen(mat, r, found_solution):
#     if r == len(mat):
#         printSolution(mat)
#         found_solution[0] = True
#         return
#     for i in range(len(mat)):
#         if not found_solution[0] and isSafe(mat, r, i):
#             mat[r][i] = 'Q'
#             nQueen(mat, r + 1, found_solution)
#             mat[r][i] = '-'

# if __name__ == '__main__':
#     N = int(input())
#     mat = [['-' for x in range(N)] for y in range(N)]
#     found_solution = [False]
#     nQueen(mat, 0, found_solution)
    
#     if not found_solution[0]:
#         print("No solution found.")
