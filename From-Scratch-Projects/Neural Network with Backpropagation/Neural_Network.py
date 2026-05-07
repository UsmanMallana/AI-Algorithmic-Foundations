import numpy as np
import copy
from image_preprocessing import load_images_from_folder, encode_labels, preprocess_image
from sklearn.model_selection import train_test_split

def relu_derivative(dA,x):
    mask = (x>0).astype(float)
    r = dA * mask
    return r

def sigmoid_derivative(dA,x):
    s = x * (1 - x)
    s = np.multiply(s,dA)
    return s

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    cache = s
    return s, cache

def relu(x):
    cache = x.copy()
    a = np.maximum(0, x)
    return a, cache

def initialize_parameters(layers_dimensions):
    parameters = {}
    L = len(layers_dimensions)

    for l in range(1,L):
        parameters['W'+str(l)] = np.random.randn(layers_dimensions[l],layers_dimensions[l-1])*0.01
        parameters['b'+str(l)] = np.zeros((layers_dimensions[l],1))

        assert parameters['W'+str(l)].shape == (layers_dimensions[l],layers_dimensions[l-1])
        assert parameters['b'+str(l)].shape == (layers_dimensions[l],1)
    return parameters

def linear_forward(A,W,b):
    Z = np.dot(W,A)+b
    cache = (A,W,b)
    return Z, cache

def linear_activation_forward(A_prev,W,b,activation):
    if activation=='relu':
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)
    elif activation=='sigmoid':
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = sigmoid(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X,parameters):
    A = X
    L = len(parameters)//2
    caches = []

    for l in range(1,L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],'relu')
        caches.append(cache)
    AL, cache = linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],'sigmoid')
    caches.append(cache)
    return AL, caches

def compute_cost(AL,Y):
    m = Y.shape[1]
    epsilon = 1e-10  # Small value to avoid log(0)
    AL = np.clip(AL, epsilon, 1 - epsilon)  # Restrict AL between epsilon and 1-epsilon
    cost = -(1/m)*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))
    cost = np.squeeze(cost)
    return cost

def linear_backward(dZ,cache):
    A_prev, W, _ = cache
    m = A_prev.shape[1]
    dW = 1/m*(np.dot(dZ,A_prev.T))
    db = 1/m*(np.sum(dZ,axis=1,keepdims=True))
    dA_prev = np.dot(W.T,dZ)
    return dA_prev, dW, db 

def linear_activation_backward(dA,cache,activation):
    linear_cache, activation_cache = cache
    if activation=='relu':
        dZ = relu_derivative(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
    elif activation=='sigmoid':
        dZ = sigmoid_derivative(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL,Y,caches):
    grads = {}
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    L = len(caches)
    epsilon = 1e-10
    dAL = -(np.divide(Y, np.clip(AL, epsilon, 1 - epsilon)) - np.divide(1 - Y, np.clip(1 - AL, epsilon, 1 - epsilon)))
    current_cache = caches[L-1]
    dA_prev, dW, db = linear_activation_backward(dAL,current_cache,'sigmoid')
    grads['dW'+str(L)] = dW
    grads['db'+str(L)] = db
    grads['dA'+str(L-1)] = dA_prev

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(grads['dA'+str(l+1)],current_cache,'relu')
        grads['dW'+str(l+1)] = dW
        grads['db'+str(l+1)] = db
        grads['dA'+str(l)] = dA_prev
    return grads

def update_parameters(params, grads, learning_rate):
    parameters = copy.deepcopy(params)
    L = len(parameters)//2

    for l in range(L):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - (learning_rate* grads['dW'+str(l+1)])
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - (learning_rate* grads['db'+str(l+1)])
    return parameters

def random_mini_batches(images, labels, batch_size=32, seed=42):
    np.random.seed(seed)
    m = images.shape[0]
    permutation = np.random.permutation(m)
    shuffled_images = images[permutation]
    shuffled_labels = labels[permutation]

    mini_batches = []
    num_complete_batches = m // batch_size

    for k in range(num_complete_batches):
        batch_images = shuffled_images[k*batch_size : (k+1)*batch_size]
        batch_labels = shuffled_labels[k*batch_size : (k+1)*batch_size]
        mini_batches.append((batch_images, batch_labels))

    if m % batch_size != 0:
        batch_images = shuffled_images[num_complete_batches*batch_size : m]
        batch_labels = shuffled_labels[num_complete_batches*batch_size : m]
        mini_batches.append((batch_images, batch_labels))

    return mini_batches

def initialize_velocity(parameters):
    L = len(parameters)//2
    v = {}

    for l in range(1,L+1):
        v['dW'+str(l)] = np.zeros(parameters['W'+str(l)].shape)
        v['db'+str(l)] = np.zeros(parameters['b'+str(l)].shape)
    return v

def update_parameters_with_momentum(parameters,grads,v,beta,learning_rate):
    L = len(parameters)//2
    for l in range(1,L+1):
        v['dW'+str(l)] = beta*v['dW'+str(l)]+(1-beta)*grads['dW'+str(l)]
        v['db'+str(l)] = beta*v['db'+str(l)]+(1-beta)*grads['db'+str(l)]
        parameters['W'+str(l)] = parameters['W'+str(l)]-(learning_rate*v['dW'+str(l)])
        parameters['b'+str(l)] = parameters['b'+str(l)]-(learning_rate*v['db'+str(l)])
    return parameters, v

def initialize_adam(parameters):
    L = len(parameters)//2
    v = {}
    s = {}
    for l in range(1,L+1):
        v['dW'+str(l)] = np.zeros(parameters['W'+str(l)].shape)
        v['db'+str(l)] = np.zeros(parameters['b'+str(l)].shape)
        s['dW'+str(l)] = np.zeros(parameters['W'+str(l)].shape)
        s['db'+str(l)] = np.zeros(parameters['b'+str(l)].shape)
    return v, s

def update_parameters_with_adam(parameters,grads,v,s,t,learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8):
    L = len(parameters)//2
    v_corrected = {}
    s_corrected = {}

    for l in range(1,L+1):
        v['dW'+str(l)] = beta1*v['dW'+str(l)]+(1-beta1)*grads['dW'+str(l)]
        v['db'+str(l)] = beta1*v['db'+str(l)]+(1-beta1)*grads['db'+str(l)]
        v_corrected['dW'+str(l)] = v['dW'+str(l)]/(1-beta1**t)
        v_corrected['db'+str(l)] = v['db'+str(l)]/(1-beta1**t)

        s['dW'+str(l)] = beta2*s['dW'+str(l)]+(1-beta2)*(grads['dW'+str(l)]**2)
        s['db'+str(l)] = beta2*s['db'+str(l)]+(1-beta2)*(grads['db'+str(l)]**2)
        s_corrected['dW'+str(l)] = s['dW'+str(l)]/(1-beta2**t)
        s_corrected['db'+str(l)] = s['db'+str(l)]/(1-beta2**t)

        parameters['W'+str(l)] = parameters['W'+str(l)]-(learning_rate*(v_corrected['dW'+str(l)]/(np.sqrt(s_corrected['dW'+str(l)])+epsilon)))
        parameters['b'+str(l)] = parameters['b'+str(l)]-(learning_rate*(v_corrected['db'+str(l)]/(np.sqrt(s_corrected['db'+str(l)])+epsilon)))
    return parameters, v, s, v_corrected, s_corrected

def model(root_dir, layer_dims, target_size, train_size, optimizer, learning_rate=0.001, num_epochs=5, batch_size=32, print_cost=True):
    np.random.seed(1)
    costs = []
    t = 1  # Adam time step
    images, labels = load_images_from_folder(root_dir, target_size=target_size)
    labels_encoded, class_names = encode_labels(labels)
    X_train, X_test, Y_train, Y_test = train_test_split(
        images, labels_encoded, test_size=train_size-1, random_state=42
    )
    layer_dims.insert(0,X_train.shape[1]*X_train.shape[2]*3)
    layer_dims.append(len(class_names))

    parameters = initialize_parameters(layer_dims)
    
    # Initialize optimizer variables
    if optimizer == 'adam':
        v, s = initialize_adam(parameters)
    elif optimizer == 'momentum':
        v = initialize_velocity(parameters)
    
    for i in range(num_epochs):
        # Create mini-batches
        mini_batches = random_mini_batches(X_train, Y_train, batch_size)
        epoch_cost = 0  # Accumulate cost for the epoch
        
        for batch in mini_batches:
            X_batch, Y_batch = batch
            
            # Forward pass on the mini-batch
            AL, caches = L_model_forward(X_batch, parameters)
            
            # Compute cost (optional: accumulate for reporting)
            batch_cost = compute_cost(AL, Y_batch)
            epoch_cost += batch_cost  # Average cost across batches
            
            # Backward pass
            grads = L_model_backward(AL, Y_batch, caches)
            
            # Update parameters
            if optimizer == 'adam':
                parameters, v, s, _, _ = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate)
                t += 1  # Increment Adam step once per batch
            elif optimizer == 'momentum':
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta=0.9, learning_rate=learning_rate)
            elif optimizer == 'gd':
                parameters = update_parameters(parameters, grads, learning_rate=learning_rate)
        
        # Compute epoch cost (average over batches)
        epoch_cost /= len(mini_batches)
        costs.append(epoch_cost)
        
        if print_cost:
            print(f"Cost after epoch {i}: {np.squeeze(epoch_cost):.6f}")
    
    return parameters, costs

