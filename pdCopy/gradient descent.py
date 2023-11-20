def objective_function(x):
    return (x + 3) ** 2

def gradient(x):
    return 2 * (x + 3)

learning_rate = 0.1
num_iterations = 100

x = 2

for i in range(num_iterations):
    grad = gradient(x)

    x = x - learning_rate * grad

    print(f"Iteration {i + 1}: x = {x}, y = {objective_function(x)}")

print("\nLocal minimum:")
print(f"x = {x}, y = {objective_function(x)}")
