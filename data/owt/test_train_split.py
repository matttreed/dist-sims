import numpy as np

TRAIN_SPLIT = 0.01
TEST_SPLIT = 0.0001

if __name__ == "__main__":
    # Read the binary file as a uint16 array
    with open("./openwebtext.bin", "rb") as f:
        data = np.fromfile(f, dtype=np.uint16)

    # Calculate the split indices
    total_elements = data.shape[0]
    train_elements = int(total_elements * TRAIN_SPLIT)
    test_elements = int(total_elements * TEST_SPLIT)

    # Split the data
    data_x = data[:train_elements]
    data_y = data[train_elements : train_elements + test_elements]

    # Save the new binary files
    data_x.tofile("owt_train.bin")
    data_y.tofile("owt_test.bin")
