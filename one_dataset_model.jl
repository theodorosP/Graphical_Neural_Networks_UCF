using Flux
using Plots
using Random
#using Statistics
using PyCall
using OneHotArrays
using Statistics
using Random

np = pyimport("numpy")

#Generating my random data
#n is the number of rows in the columns in the matrix we want to create

function generate_real_data(n)
  Random.seed!(1)
  x1 = rand(1, n) .- 0.5
  x2 = (x1 .* x1 ) * 3 .+ rand(1, n) * 0.5 
  final = vcat(x1, x2)
  return final
end  

function generate_fake_data(n)
  Random.seed!(1)
  th  = 2*π*rand(1,n)
  r  = rand(1,n)/3
  x1 = @. r * cos(th)
  x2 = @. r*sin(th) + 0.5
  final = vcat(x1, x2)
return final 
end

#This function splits radomply the dataset, X and Y matrix, in training and testing sets
#data is the matrix we want to split
#at is the percentage we want to split. If we want to split in 70-30, then at = 0.7 

function split_matrix(data, at)
    Random.seed!(1)
    n =size(data)[2]
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    trained = data[:,train_idx]
    tested = data[:,test_idx]
    return(trained, tested)
end


#This function in our neural network function
#x and y are the input matrices
function NeuralNetwork(x, y)
return Chain(
             Dense(size(x, 1), 25, relu),
             Dense(25, size(y, 1)), sigmoid),
             Flux.flatten
             )
end

#This function in our neural network function
#x and y are the input matrices
function NeuralNetwork(x, y)
return Chain(
             Dense(size(x, 1), 25, relu),
             Dense(25, size(y, 1)), sigmoid),
             Flux.flatten
             )
end


train_size = 10000
real_data = generate_real_data(train_size)
fake_data = generate_fake_data(train_size);

scatter(real_data[1,1:train_size], real_data[2,1:train_size], label = "Real Data")
scatter(fake_data[1,1:train_size], fake_data[2,1:train_size], label = "Fake Data", color = "orange")

scatter(real_data[1,1:train_size], real_data[2,1:train_size], label = "Real Data")
scatter!(fake_data[1,1: train_size], fake_data[2,1:train_size], label = "Fake Data")

X = hcat(real_data, fake_data)
Y = vcat(ones(train_size),zeros(train_size))

#Y_aux = vcat(ones(100),zeros(100))
#Y=Y_aux[:,:]

train_x, test_x = split_matrix(X, 0.66)
train_y, test_y = split_matrix(Y', 0.66)

#check the dimentions of training and testing dataset
println("X", size(X))
println("Y'", size(Y'))
println("train_x", size(train_x))
println("train_y", size(train_y))
println("test_x", size(test_x))
println("test_y", size(test_y))


data = Flux.Data.DataLoader((X, Y'), batchsize=100,shuffle=true)
m = NeuralNetwork(train_x, train_y)
opt = Descent(0.05)
loss(x, y) = sum(Flux.Losses.binarycrossentropy(m(x), y))

epochs = Int64[]
loss_on_train = Float32[]
loss_on_test = Float32[]

ps = Flux.params(m)
epoch = 500
for i in 1:epoch
    Flux.train!(loss, ps, data, opt)
    push!(epochs, i)
    push!(loss_on_test, loss(test_x, test_y))
    push!(loss_on_train, loss(train_x, train_y))
end
println(mean(m(real_data)), mean(m(fake_data))) # Print model prediction

plot(epochs, loss_on_train, lab="Training loss", c=:black, lw=2);
plot!(epochs, loss_on_test, lab="Validation loss", c=:teal, ls=:dot);
yaxis!("Loss", :log);
xaxis!("Training epoch")

scatter(real_data[1,1:train_size],real_data[2,1:train_size],zcolor=m(real_data)')
scatter!(fake_data[1,1:train_size],fake_data[2,1:train_size],zcolor=m(fake_data)',legend=false)

#predictions for test_x data
y = m(test_x)


resDict = Dict()
resDict["model"] = m
params = Flux.params(m);
resDict["params"] = params
println(resDict["model"].layers);


#test one onehotencoding with my function 
X = hcat(real_data, fake_data)


l = []
for i in 1:train_size
  append!(l, 0)
end 

Y_ = vcat(ones(train_size),l)

Y = one_hot_encode_matrix(Y_, 2)


train_x, test_x = split_matrix(X, 0.66)
train_y, test_y = split_matrix(Y', 0.66)

#check the dimentions of training and testing dataset
println("size(X): ", size(X))
println("size(Y): ", size(Y))
println("size(train_x): ", size(train_x))
println("size(train_y): ", size(train_y))
println("size(test_x): ", size(test_x))
println("size(test_y): ", size(test_y))





data = Flux.Data.DataLoader((train_x, train_y), batchsize=100,shuffle=true)
m = NeuralNetwork(train_x, train_y)
opt = Descent(0.05)
loss(x, y) = sum(Flux.Losses.binarycrossentropy(m(x), y))


epochs = Int64[]
loss_on_train = Float32[]
loss_on_test = Float32[]

ps = Flux.params(m)
epoch = 1000
for i in 1:epoch
    Flux.train!(loss, ps, data, opt)
    push!(epochs, i)
    push!(loss_on_test, loss(test_x, test_y))
    push!(loss_on_train, loss(train_x, train_y))
end
println(mean(m(real_data)[1, 1:end]), mean(m(fake_data)[1, 1:end])) # Print model prediction

plot(epochs, loss_on_train, lab="Training loss", c=:black, lw=2);
plot!(epochs, loss_on_test, lab="Validation loss", c=:teal, ls=:dot);
yaxis!("Loss", :log);
xaxis!("Training epoch")

scatter(real_data[1,1:train_size],real_data[2,1:train_size],zcolor=m(real_data)')
scatter!(fake_data[1,1:train_size],fake_data[2,1:train_size],zcolor=m(fake_data)',legend=false)

y = onecold(test_y)
y_hat = m(test_x)
y_hat = onecold(y_hat)
#for i in 1:length(y_hat)
#  println("actual: ", onecold(test_y)[i], " pred: ", y_hat[i])
#end 

s = 0
for i in 1:length(y_hat)
  if y_hat[i] == y[i]
    s += 1
  end 
end  
println("Accuracy: ", round(s*100/length(y_hat), digits = 2), "%")

