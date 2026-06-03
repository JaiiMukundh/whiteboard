import mnist_loader
import network

print("Loading MNIST dataset...")
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

print("Initializing Neural Network with 30 hidden neurons...")
net = network.Network([784, 200, 10])

print("Beginning Training: performing 30 Epochs")

# Stage 1: Big steps to get down the mountain quickly (Epochs 0-10)
print("--- Stage 1: eta = 0.1 ---")
net.SGD(training_data, epochs=10, mini_batch_size=10, eta=0.1, lmbda=2.5, test_data=test_data)

# Stage 2: Medium steps to navigate the valley floor (Epochs 11-20)
print("--- Stage 2: eta = 0.01 ---")
net.SGD(training_data, epochs=10, mini_batch_size=10, eta=0.01, lmbda=5.0, test_data=test_data)

# Stage 3: Tiny steps to settle at the absolute minimum and stop the wiggle/overfitting (Epochs 21-30)
print("--- Stage 3: eta = 0.001 ---")
net.SGD(training_data, epochs=10, mini_batch_size=10, eta=0.001, lmbda=5.0, test_data=test_data)