import heapq
class node:
	def __init__(self, freq, symbol, left=None, right=None):
		self.freq = freq
		self.symbol = symbol
		self.left = left
		self.right = right
		self.huff = ''
	def __lt__(self, nxt):
		return self.freq < nxt.freq

def printNodes(node, val=''):
	newVal = val + str(node.huff)
	if(node.left):
		printNodes(node.left, newVal)
	if(node.right):
		printNodes(node.right, newVal)
	if(not node.left and not node.right):
		print(f"{node.symbol} -> {newVal}")
  
chars = ['a', 'b', 'c', 'd', 'e', 'f']
freq = [ 5, 9, 12, 13, 16, 45]
nodesList = []

for i in range(len(chars)):
	heapq.heappush(nodesList, node(freq[i], chars[i]))
 
while len(nodesList) > 1:
	left = heapq.heappop(nodesList)
	right = heapq.heappop(nodesList)
	left.huff = 0
	right.huff = 1
	newNode = node(left.freq+right.freq, left.symbol+right.symbol, left, right)
	heapq.heappush(nodesList, newNode)
 
printNodes(nodesList[0])
