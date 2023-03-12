using Flux, Statistics, MLDatasets, DataFrames, Printf

# initializing the dataset.
dataset = BostonHousing();
x, y = BostonHousing(as_df=false)[:];
x, y = Float32.(x), Float32.(y);

# Split the data into training and testing data
x_train, x_test, y_train, y_test = x[:, 1:400], x[:, 401:end], y[:, 1:400], y[:, 401:end];

x_train |> size, x_test |> size, y_train |> size, y_test |> size

# Normalize the training data
x_train_n = Flux.normalise(x_train);

# Build a Flux linear regression model
model = Dense(13 => 1)

# define the loss function
function loss(model, x, y)
    ŷ = model(x)
    Flux.mse(ŷ, y)
end;

# Train the Flux model
function train_model()
    dLdm, _, _ = gradient(loss, model, x_train_n, y_train)
    @. model.weight = model.weight - 0.000001 * dLdm.weight
    @. model.bias = model.bias - 0.000001 * dLdm.bias
end;

tolerance = 1e-4
max_iter = 100000
global iter = 1

while iter <= max_iter
    local loss_init = loss(model, x_train_n, y_train)
    train_model()
    new_loss = loss(model, x_train_n, y_train)
    if abs(loss_init - new_loss) < tolerance
        break
    else
        loss_init = new_loss
        @printf("Iteration %d: loss = %f\n", iter, new_loss)
        global iter += 1
    end
end

# Test the trained model
# normalize the test data
x_test_n = Flux.normalise(x_test)

# predict the data
loss(model, x_test_n, y_test)
