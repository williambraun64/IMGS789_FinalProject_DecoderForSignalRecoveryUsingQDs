"""
Created on Sun Nov 10 20:33:46 2024

@author: willyb
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

# Directory containing the .txt files
directory_path = r'C:\Users\willb\QDFF\ecospeclib_1727922338546\ecospeclib-all'

# Dictionary to store data
data_dict = {}

# Function to read data from each file with error handling for encoding issues
def read_data_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='ISO-8859-1') as f:
            lines = f.readlines()

    # Initialize variables
    name = None
    x_data = []
    y_data = []

    # Parse the file content
    for line in lines:
        # Check if line starts with 'Name' to get the name
        if line.startswith("Name:"):
            name = line.split(":", 1)[1].strip()
        
        # Check if line contains numeric data for X and Y
        elif re.match(r"^\s*\d+(\.\d+)?\s+\d+(\.\d+)?\s*$", line):
            x, y = map(float, line.split())
            x_data.append(x)
            y_data.append(y)

    return name, x_data, y_data

# Function to plot all spectra on the same plot
def plot_spectra(data_dict):
    plt.figure(figsize=(10, 6))
    for name, data in data_dict.items():
        plt.plot(data['x'], data['y'], label=name)

    plt.xlabel('Wavelength (micrometers)')
    plt.ylabel('Reflectance (percent)')
    plt.title('Spectral Data')
    #plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Process each .txt file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory_path, filename)
        name, x_data, y_data = read_data_from_file(file_path)

        # Check if name already exists in the dictionary and make unique if necessary
        original_name = name
        count = 1
        while name in data_dict:
            name = f"{original_name}_{count}"
            count += 1
        
        # Store in dictionary
        data_dict[name] = {'x': x_data, 'y': y_data}

# Plot all spectra
plot_spectra(data_dict)

# %%


# Spectra go from 0.35 to 2.5 microns with 2151 entries


# Parameters
number_of_curves = 60
K = number_of_curves
start_wavelength = 350
end_wavelength = 2500
sigma = 100
height = 17000
scale_factor = 10
step_up = 0.1
specified_slope = -0.01  # slope of linear part
noise_level = 64  # noise level
slope_sub_per_rep = 0.00005

# Define the range of x values with 2151 entries
x = np.linspace(start_wavelength, end_wavelength, 2151)

# Values for Gaussian centers, adjusted to the new x range
mu_vec = np.linspace(start_wavelength + 3 * sigma, end_wavelength - 3 * sigma, number_of_curves)

# Define the Gaussian function with specified height
def gaussian(x, mu, sigma, height):
    return height * np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

# Initialize variables
absorbance_matrix = []

# Step-up array for each curve
step_up_vec = step_up * np.ones_like(x)

# Generate curves
for i in range(number_of_curves):
    mu = mu_vec[i]
    specified_slope -= i * slope_sub_per_rep

    # Compute the Gaussian values over the specified range
    y_gaussian = gaussian(x, mu, sigma, height)

    # Find intersection point of the linear function and Gaussian curve
    x_intersect = mu - (3 * sigma)
    y_intersect = y_gaussian[np.where(x <= x_intersect)[0][-1]]

    # Define linear function
    def linear(x_val):
        return specified_slope * (x_val - x_intersect) + y_intersect

    # Combine Gaussian and linear parts
    y_combined = np.copy(y_gaussian)
    left_indices = x <= x_intersect
    y_combined[left_indices] = linear(x[left_indices])

    # Add Gaussian noise
    y_noisy = y_combined + noise_level * np.random.randn(*y_combined.shape)

    # Smooth the noisy curve
    smoothness = 0.05  # proportion of range to smooth
    smooth_window = int(smoothness * len(x))
    y_smoothed = gaussian_filter1d(y_noisy, smooth_window)

    # Apply step up for each curve
    y_smoothed_stepped = y_smoothed + i * step_up_vec

    # Append to absorbance matrix
    absorbance_matrix.append(y_smoothed_stepped)

# Convert to numpy array for normalization
absorbance_matrix = np.array(absorbance_matrix)

# Normalize the absorbance curves
normalization_value = np.max(absorbance_matrix)
absorbance_matrix /= normalization_value

# Plot all normalized absorbance curves
plt.figure(figsize=(10, 6))
for i in range(number_of_curves):
    plt.plot(x, absorbance_matrix[i, :], linewidth=1)

plt.title('Normalized Quantum Dot Absorbance Curves')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance (a.u.)')
plt.grid(True)
plt.show()

# Create measurement matrix
measurement_matrix = 1 - absorbance_matrix


# %% Remove empty arrays from data_dict

def clean_data_dict(data_dict):
    # Filter out entries where 'x' or 'y' are empty
    cleaned_dict = {key: value for key, value in data_dict.items() if value['x'] and value['y']}
    return cleaned_dict

# Clean the data_dict
data_dict = clean_data_dict(data_dict)


# %% Find most common length 

from collections import Counter

# Step 1: Extract the lengths of spectrum['y']
y_lengths = [len(spectrum['y']) for spectrum in data_dict.values()]

# Step 2: Use Counter to count the frequency of each length
length_counts = Counter(y_lengths)

# Step 3: Find the most common length
most_common_length, most_common_count = length_counts.most_common(1)[0]

# Step 4: Print the result
print(f"The most common length is {most_common_length} with {most_common_count} occurrences.")

# %% Create new data_dict with only the specta that are the same length. 

# Step 2: Create a new dictionary with only the entries that have the longest 'y' array
data_dict_sim = {name: spectrum for name, spectrum in data_dict.items() if len(spectrum['y']) == most_common_length}

#Now, need to normalize the entries


#This also adds a zero to ends of x and y to make even size
def normalize_y_values(data_dict):
  """Normalizes the 'y' values in each entry of the given dictionary using min-max normalization,
  and adds a zero to the end of the 'x' and 'y' lists.

  Args:
    data_dict: A dictionary where each value is a dictionary with 'x' and 'y' keys.

  Returns:
    A new dictionary with the normalized 'y' values and extended 'x' and 'y' lists.
  """

  normalized_data = {}
  for key, value in data_dict.items():
    x_values = value['x']
    y_values = value['y']

    y_min = min(y_values)
    y_max = max(y_values)

    normalized_y = [(y - y_min) / (y_max - y_min) for y in y_values]
    normalized_data[key] = {'x': x_values, 'y': normalized_y}

  return normalized_data

data_dict_sim_norm = normalize_y_values(data_dict_sim)

#%% ADDDD
from scipy.ndimage import gaussian_filter1d

def simulate_optical_spectrum(n, noise_level=0, smoothness=100):
    """
    Simulates an optical spectrum of length n, normalized between 0 and 1.

    Parameters:
        n (int): Length of the spectrum.
        noise_level (float): Amplitude of noise to add to the spectrum.
        smoothness (int): Controls the smoothness of the spectrum.

    Returns:
        np.ndarray: Simulated optical spectrum with values between 0 and 1.
    """
    # Generate a random base spectrum
    spectrum = np.random.rand(n)
    
    # Smooth the spectrum to emulate natural curves
    spectrum = gaussian_filter1d(spectrum, sigma=smoothness)
    
    # Add random noise
    noise = np.random.normal(0, noise_level, n)
    spectrum += noise
    
    # Normalize to ensure min is 0 and max is 1
    spectrum -= np.min(spectrum)  # Shift to ensure min is 0
    spectrum /= np.max(spectrum)  # Scale to ensure max is 1
    
    return spectrum

def add_inverted_spectra(data_dict):
    """
    Accesses spectra in the dictionary, inverts them, and appends them
    to the end of the dictionary with new keys.

    Parameters:
        data_dict (dict): Dictionary containing spectra with 'x' and 'y' values.

    Returns:
        None: Modifies the dictionary in place by adding inverted spectra.
    """
    inverted_count = 1  # Counter for naming inverted keys

    for key in list(data_dict.keys()):
        # Access the current spectrum
        original_x = data_dict[key]['x']
        original_y = np.array(data_dict[key]['y'])  # Convert to a NumPy array if not already

        # Invert the spectrum
        inverted_y = 1 - original_y

        # Add inverted spectrum to the dictionary
        new_key = f"THEInverted{inverted_count}"
        data_dict[new_key] = {'x': original_x, 'y': inverted_y.tolist()}  # Convert back to list if needed

        inverted_count += 1  # Increment the counter
        print(inverted_count)
        
    return inverted_count


# # Simulated spectra show
# n = 2151 # Length of the spectrum
# simulated_spectrum = simulate_optical_spectrum(n)

# plt.plot(simulated_spectrum)
# plt.title("Simulated Optical Spectrum")
# plt.xlabel("Wavelength (arbitrary units)")
# plt.ylabel("Intensity (normalized)")
# plt.show()


# Invert all spectra and add to dict to avoid overfitting
inv_count = add_inverted_spectra(data_dict_sim_norm)

#%%

# User-specified parameters
num_simulations = 614# Number of random spectra simulations to add
n = 2151          # Length of the spectrum

# Simulate and append new data to the existing dictionary
for i in range(1, num_simulations + 1):
    key = f"Sim{i}"  # Create keys like 'Sim1', 'Sim2', etc.
    
    # Generate 'x' and 'y'
    x = np.linspace(350, 2500, n)  # Linear space for 'x'
    y = simulate_optical_spectrum(n)  # Simulated spectrum for 'y'

    plt.figure()
    if 0 < i < 10:
        plt.plot(x, y)
        plt.show()

    plt.show()
    
    # Append to the existing dictionary
    data_dict_sim_norm[key] = {'x': x, 'y': y}


 # %% Compute dot product for all spectra that have same length that is most common
 #You should also make sure this is the length of simualted QD abosrbance curves and the consequent measurement matrix entries


all_dot_product_results = []

for name_, spectrum_ in data_dict_sim_norm.items():
    dot_product_results = []
    dot_product_results.append(name_)
    
    spectrum_y = spectrum_['y']
    
    for i in range(len(measurement_matrix)):
        QD_spec_meas = measurement_matrix[i]
    
        dot_product = np.dot(QD_spec_meas, spectrum_y)
        dot_product_results.append(dot_product)
        
    all_dot_product_results.append(dot_product_results)

# %%
#Normalize!! min max normalization

all_dot_product_results_norm = []

for p in range(len(all_dot_product_results)):
    current_result_row = all_dot_product_results[p][1:]

    # Min-Max normalization
    norm_result = (current_result_row - np.min(current_result_row)) / (np.max(current_result_row) - np.min(current_result_row))

    # Create a new list by appending the first element to the normalized result
    norm_result_labeled = [all_dot_product_results[p][0]] + list(norm_result)

    all_dot_product_results_norm.append(norm_result_labeled)


# %% Now all the dot_product results can be used in a VAE!


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt  # For plotting

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Decoder(nn.Module):
    def __init__(self, latent_dims, output_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 538)
        self.linear2 = nn.Linear(538, 1076)
        self.linear3 = nn.Linear(1076, output_dims)

        # Call the weight initialization method
        self.initialize_weights()

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = self.linear3(z)
        return z

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# Reconstruction loss function
def reconstruction_loss(output, target):
    return F.mse_loss(output, target)

# %% Now divide into training and test sets

#But if we were to just slice, there would be a section of regular spectra, a seciton of the inverted spectra, and a section of the random spectra.
#So, need to shuffle the entries in all_dot and data_dict

import random
import math

def shuffle_dict_and_list(data_dict, data_list):
    """
    Randomly shuffles a dictionary and a list, preserving correspondence between their entries.

    Parameters:
        data_dict (dict): The dictionary to shuffle (keys and values are preserved together).
        data_list (list): The list to shuffle, corresponding to the dictionary entries.

    Returns:
        tuple: A shuffled dictionary and list.
    """
    if len(data_dict) != len(data_list):
        raise ValueError("The dictionary and list must have the same number of entries.")

    # Extract keys, values, and corresponding list items as a single structure
    combined = list(zip(data_dict.items(), data_list))
    
    # Shuffle the combined structure
    random.shuffle(combined)
    
    # Reconstruct the dictionary and list from the shuffled pairs
    shuffled_dict = {key: value for (key, value), _ in combined}
    shuffled_list = [list_item for _, list_item in combined]
    
    return shuffled_dict, shuffled_list

# Shuffle
shuffled_dict, shuffled_dots = shuffle_dict_and_list(data_dict_sim_norm, all_dot_product_results_norm)



# %% Introduce discrete cosine transform (do for both measurement and spectra or just spectra?)
from scipy.fft import dct
from scipy.fft import idct

# Apply DCT to the list
dct_shuffled_dots = []
for list_ in shuffled_dots:
    dct_transformed_dot = []
    dct_transformed_dot.append(list_[0])
    dct_transformed_dot_unlabeled = dct(list_[1:], type=2, norm='ortho')
    for entry in  dct_transformed_dot_unlabeled:
        dct_transformed_dot.append(entry)
    dct_shuffled_dots.append(dct_transformed_dot)

dct_shuffled_dict = {}

for key in shuffled_dict:
    original_x = shuffled_dict[key]['x']
    original_y = shuffled_dict[key]['y']
    
    # Apply DCT to 'y'
    dct_y = dct(original_y, type=2, norm='ortho')  # Apply DCT

    # Add to new dictionary
    dct_shuffled_dict[key] = {"x": original_x, "y": dct_y.tolist()}  # Ensure it's a list for consistency
    
    

#%% Create training set and test set using dct

half_length = math.floor(len(dct_shuffled_dots)/2)

training_dct_shuffled_dots = dct_shuffled_dots[half_length:]
training_dct_shuffled_dict = dict(list(dct_shuffled_dict.items())[half_length:])

test_dct_shuffled_dots = dct_shuffled_dots[:half_length]
test_dct_shuffled_dict = dict(list(dct_shuffled_dict.items())[:half_length])

# %%creat training and test sets
half_length = math.floor(len(shuffled_dots)/2)

training_dot_product_results_norm = shuffled_dots[half_length:]
training_data_dict_sim_norm = dict(list(shuffled_dict.items())[half_length:])

test_dot_product_results_norm = shuffled_dots[:half_length]
test_data_dict_sim_norm = dict(list(shuffled_dict.items())[:half_length])

#%%
import torch
import torch.optim as optim



def train_decoder(decoder, optimizer, reconstruction_loss, latent_dims, latent_vector_array, target_spectra_dictionary, test_latent_vector_array, test_target_spectra_dictionary, device, epochs, batch_size):
    """
    Train the decoder model and evaluate on test data.

    Args:
        decoder (torch.nn.Module): The decoder model to train.
        optimizer (torch.optim.Optimizer): Optimizer for the decoder.
        reconstruction_loss (callable): Loss function for training.
        latent_dims (int): Size of the latent space.
        latent_vector_array (list): List of training latent vectors and corresponding names.
        target_spectra_dictionary (dict): Dictionary of corresponding 'x' and 'y' values for training names.
        test_latent_vector_array (list): List of test latent vectors and corresponding names.
        test_target_spectra_dictionary (dict): Dictionary of corresponding 'x' and 'y' values for test names.
        device (torch.device): Device to use for computation (e.g., 'cpu' or 'cuda').
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        dict: Dictionary containing loss history across epochs for both training and test data.
    """
    # List to store loss history
    loss_history = {'train_loss': [], 'test_loss': []}

    # Training loop
    for epoch in range(epochs):
        total_train_loss = 0
        total_test_loss = 0
        
        train_data = []
        target_data = []
        
        # Collect training data
        for row in latent_vector_array:
            name = row[0]  # The first entry is the name
            latent_vector = torch.tensor(row[1:], dtype=torch.float32)  # Remaining are the latent vector
            target_vector = torch.tensor(target_spectra_dictionary[name]['y'], dtype=torch.float32)  # Get the target 'y' value
            
            # Ensure latent_vector size matches latent_dims
            if latent_vector.size(0) != latent_dims:
                raise ValueError(f"Latent vector size must be {latent_dims}, but got {latent_vector.size(0)}")
            
            # Append to train_data and target_data
            train_data.append(latent_vector)
            target_data.append(target_vector)

        # Convert to tensors
        train_data = torch.stack(train_data)
        target_data = torch.stack(target_data)

        # Create a DataLoader for batch processing
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_data, target_data),
            batch_size=batch_size, shuffle=True
        )

        # Train for one epoch
        decoder.train()  # Set model to training mode
        for batch_idx, (latent, target) in enumerate(train_loader):
            latent = latent.to(device)
            target = target.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass through the decoder
            output = decoder(latent)
            
            # Compute the reconstruction loss
            loss = reconstruction_loss(output, target)
            
            # Backpropagate the loss
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            total_train_loss += loss.item()

        # Save the training loss for this epoch
        loss_history['train_loss'].append(total_train_loss / len(train_loader))

        # Evaluate on test data
        decoder.eval()  # Set model to evaluation mode
        test_data = []
        test_target_data = []
        
        # Collect test data
        for row in test_latent_vector_array:
            name = row[0]  # The first entry is the name
            latent_vector = torch.tensor(row[1:], dtype=torch.float32)  # Remaining are the latent vector
            target_vector = torch.tensor(test_target_spectra_dictionary[name]['y'], dtype=torch.float32)  # Get the target 'y' value
            
            # Ensure latent_vector size matches latent_dims
            if latent_vector.size(0) != latent_dims:
                raise ValueError(f"Latent vector size must be {latent_dims}, but got {latent_vector.size(0)}")
            
            # Append to test_data and test_target_data
            test_data.append(latent_vector)
            test_target_data.append(target_vector)

        # Convert to tensors
        test_data = torch.stack(test_data)
        test_target_data = torch.stack(test_target_data)

        # Create a DataLoader for test data
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_data, test_target_data),
            batch_size=batch_size, shuffle=False
        )

        with torch.no_grad():
            for batch_idx, (latent, target) in enumerate(test_loader):
                latent = latent.to(device)
                target = target.to(device)

                # Forward pass through the decoder
                output = decoder(latent)

                # Compute the reconstruction loss
                loss = reconstruction_loss(output, target)
                total_test_loss += loss.item()

        # Save the test loss for this epoch
        loss_history['test_loss'].append(total_test_loss / len(test_loader))

        # Print loss for the current epoch
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {total_train_loss / len(train_loader):.4f}, Test Loss: {total_test_loss / len(test_loader):.4f}")

    return loss_history


# Function to plot the loss curves
def plot_loss_history(loss_history):
    """
    Plots the training and test loss history.

    Args:
        loss_history (dict): Dictionary containing 'train_loss' and 'test_loss' lists.
    """
    epochs = range(1, len(loss_history['train_loss']) + 1)
    
    plt.figure(figsize=(10, 6))
    ltr = loss_history['train_loss']
    lte = loss_history['test_loss']
    plt.plot(epochs, ltr, label='Training Loss')
    plt.plot(epochs, lte, label='Test Loss', linestyle='--')
    
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training and Test Loss Over 100 Epochs', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Make the tick labels larger
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.show()


# %% Train

# Define parameters
latent_dims = 60  # number of QD spectra
output_dims = 2151  # Example output dimension
epochs = 100
batch_size = 10
learning_rate = 1e-3

# Initialize decoder and optimizer
decoder = Decoder(latent_dims, output_dims).to(device)
optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

# Train the model
loss_history = train_decoder(
    decoder=decoder,
    optimizer=optimizer,
    reconstruction_loss=reconstruction_loss,
    latent_dims=latent_dims,
    latent_vector_array=training_dot_product_results_norm,
    target_spectra_dictionary=training_data_dict_sim_norm,
    test_latent_vector_array=test_dot_product_results_norm, 
    test_target_spectra_dictionary=test_data_dict_sim_norm,
    device=device,
    epochs=epochs,
    batch_size=batch_size
)


#%%
# Plot loss curve after training
plot_loss_history(loss_history)

# %% Now use decoder to recover spectra using test set

unlabeled_test_dot_product_results_norm = []

for entry in test_dot_product_results_norm:
    unlabeled_entry = entry[1:]
    unlabeled_test_dot_product_results_norm.append(unlabeled_entry)


# Convert test_dot_product_results_norm to a tensor
test_dot_product_results_norm_tensor = torch.tensor(unlabeled_test_dot_product_results_norm, dtype=torch.float32)



import time


# Start the timer
start_time = time.time()

# Get the model's predictions (recovered data)
with torch.no_grad():
    decoder.eval()  # Set model to evaluation mode
    recovered_data = decoder(test_dot_product_results_norm_tensor)
    
# Stop the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Time to recover 921 test spectra: {elapsed_time} seconds")

    
#%% Now divide back into real, inmverted, and sim to see how it performs on each

def categorize_values_arr(array_of_lists):
    """
    Categorizes lists from an array based on the first entry of each list.

    Args:
        array_of_lists (list): An array (list) of lists to process.

    Returns:
        tuple: Three lists containing categorized sublists.
    """
    inverted_lists = []
    sim_lists = []
    other_lists = []

    for sublist in array_of_lists:
        if isinstance(sublist, list) and sublist:  # Ensure it's a non-empty list
            if str(sublist[0]).startswith('THEInverted'):
                inverted_lists.append(sublist)
            elif str(sublist[0]).startswith('Sim'):
                sim_lists.append(sublist)
            else:
                other_lists.append(sublist)

    return inverted_lists, sim_lists, other_lists

def categorize_values_dict(input_dict):
    """
    Categorizes lists from a dictionary based on the key.

    Args:
        input_dict (dict): A dictionary where keys are used for categorization.

    Returns:
        tuple: Three lists containing categorized sublists.
    """
    inverted_lists = []
    sim_lists = []
    other_lists = []

    for key, value in input_dict.items():
        
        target_spectrum = value['y']

        if key.startswith('THEInverted'):
            inverted_lists.append(target_spectrum)
        elif key.startswith('Sim'):
             sim_lists.append(target_spectrum)
        else:
            other_lists.append(target_spectrum)

    return inverted_lists, sim_lists, other_lists

# Example usage:
[inverted_latent_vecs, simulated_latent_vecs, real_latent_vecs] = categorize_values_arr(test_dot_product_results_norm)

[inverted_target_vecs, simulated_target_vecs, real_target_vecs] = categorize_values_dict(test_data_dict_sim_norm)


#%% REAL DATA

real_latent_test = []

for entry in real_latent_vecs:
    unlabeled_entry = entry[1:]
    real_latent_test.append(unlabeled_entry)


# Convert test_dot_product_results_norm to a tensor
real_latent_test_tensor = torch.tensor(real_latent_test, dtype=torch.float32)

# Get the model's predictions (recovered data)
with torch.no_grad():
    decoder.eval()  # Set model to evaluation mode
    recovered_real_data = decoder(real_latent_test_tensor)
    
Real_MSE_losses = []
reco_real_arr = []

for y in range(len(recovered_real_data)):
    reco = recovered_real_data[y]
    reco_list = reco.tolist()
    reco_real_arr.append(reco_list)
    targ = torch.tensor(real_target_vecs[y], dtype=torch.float32)
    Real_MSE_loss = reconstruction_loss(reco, targ).item()
    Real_MSE_losses.append(Real_MSE_loss)
    
real_average_loss = sum(Real_MSE_losses)/len(Real_MSE_losses)

number_of_spectra_to_look_At = 1

for t in range(number_of_spectra_to_look_At):
    x12 = np.linspace(350, 2500, 2151)
    plt.plot(x12, reco_real_arr[t])
plt.title('Recovered ECOSTRESS Spectra')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance (a.u.)')
plt.show()

for t in range(number_of_spectra_to_look_At):
    x12 = np.linspace(350, 2500, 2151)
    plt.plot(x12, real_target_vecs[t])
plt.title('Target ECOSTRESS Spectra')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance (a.u.)')
plt.show()
#%%
for t in range(number_of_spectra_to_look_At):
    x12 = np.linspace(350, 2500, 2151)
    inverted_vector = [1 - x for x in real_target_vecs[t]]
    plt.plot(x12, inverted_vector)
plt.title('Example Inverse ECOSTRESS Spectra')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance (a.u.)')
plt.show()

#%% INVERTED DATA

inverted_latent_test = []

for entry in inverted_latent_vecs:
    unlabeled_entry = entry[1:]
    inverted_latent_test.append(unlabeled_entry)


# Convert test_dot_product_results_norm to a tensor
inverted_latent_test_tensor = torch.tensor(inverted_latent_test, dtype=torch.float32)

# Get the model's predictions (recovered data)
with torch.no_grad():
    decoder.eval()  # Set model to evaluation mode
    recovered_inverted_data = decoder(inverted_latent_test_tensor)
    
inverted_MSE_losses = []
reco_inverted_arr = []

for y in range(len(recovered_real_data)):
    reco = recovered_inverted_data[y]
    reco_list = reco.tolist()
    reco_inverted_arr.append(reco_list)
    targ = torch.tensor(inverted_target_vecs[y], dtype=torch.float32)
    inverted_MSE_loss = reconstruction_loss(reco, targ).item()
    inverted_MSE_losses.append(inverted_MSE_loss)
    
inverted_average_loss = sum(inverted_MSE_losses)/len(inverted_MSE_losses)

number_of_spectra_to_look_At = 15

for t in range(number_of_spectra_to_look_At):
    x12 = np.linspace(350, 2500, 2151)
    plt.plot(x12, reco_inverted_arr[t])
plt.title('Recovered Inverted ECOSTRESS Spectra')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance (a.u.)')
plt.show()

for t in range(number_of_spectra_to_look_At):
    x12 = np.linspace(350, 2500, 2151)
    plt.plot(x12, inverted_target_vecs[t])
plt.title('Target Inverted ECOSTRESS Spectra')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance (a.u.)')
plt.show()


#%% SIMULATED DATA

simulated_latent_test = []

for entry in simulated_latent_vecs:
    unlabeled_entry = entry[1:]
    simulated_latent_test.append(unlabeled_entry)


# Convert test_dot_product_results_norm to a tensor
simulated_latent_test_tensor = torch.tensor(simulated_latent_test, dtype=torch.float32)

# Get the model's predictions (recovered data)
with torch.no_grad():
    decoder.eval()  # Set model to evaluation mode
    recovered_simulated_data = decoder(simulated_latent_test_tensor)
    
simulated_MSE_losses = []
reco_simulated_arr = []

for y in range(len(recovered_simulated_data)):
    reco = recovered_simulated_data[y]
    reco_list = reco.tolist()
    reco_simulated_arr.append(reco_list)
    targ = torch.tensor(simulated_target_vecs[y], dtype=torch.float32)
    simulated_MSE_loss = reconstruction_loss(reco, targ).item()
    simulated_MSE_losses.append(simulated_MSE_loss)
    
simulated_average_loss = sum(simulated_MSE_losses)/len(simulated_MSE_losses)

number_of_spectra_to_look_At = 1

for t in range(number_of_spectra_to_look_At):
    x12 = np.linspace(350, 2500, 2151)
    plt.plot(x12, reco_simulated_arr[t])
plt.title('Recovered Simulated Spectra')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance (a.u.)')
plt.show()

for t in range(number_of_spectra_to_look_At):
    x12 = np.linspace(350, 2500, 2151)
    plt.plot(x12, simulated_target_vecs[t])
plt.title('Target Simulated Spectra')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance (a.u.)')
plt.show()





#%% Plot test and recovered
   
number_of_spectra_to_look_At = 15

for t in range(number_of_spectra_to_look_At):
    x12 = np.linspace(350, 2500, 2151)
    plt.plot(x12, recovered_data[t])
plt.title('Recovered Test Spectra')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance (a.u.)')
plt.show()

all_original_spectra = []
for name_, spectrum_ in test_data_dict_sim_norm.items():
    
    spectrum_y_ = spectrum_['y']
    all_original_spectra.append(spectrum_y_)
    
for y in range(number_of_spectra_to_look_At):
    plt.plot(x12, all_original_spectra[y])
plt.title('Original Test Spectra')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance (a.u.)')
plt.show()

#%% Compare loss


# Convert test_dot_product_results_norm to a tensor
recovered_data_tensor = torch.tensor(recovered_data, dtype=torch.float32)
all_original_spectra_tensor = torch.tensor(all_original_spectra, dtype=torch.float32)


test_losses = []
for j in range(len(recovered_data)):
    test_loss = reconstruction_loss(recovered_data_tensor[j], all_original_spectra_tensor[j])
    test_losses.append(test_loss)
    
    
mean_reco_loss = np.mean(test_losses)

plt.plot(test_losses)


plt.axhline(y=mean_reco_loss, color='r', linestyle='--', label='y=4')

# Optional: Add labels and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Data with Horizontal Line')
plt.legend()
plt.show()
