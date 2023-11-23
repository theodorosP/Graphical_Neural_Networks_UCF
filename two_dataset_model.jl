using Flux
using Plots
using Random
#using Statistics
using PyCall
using OneHotArrays
using Statistics
using Random

#do the same with three datasets. real_data, fake_data_1, fake_data_2

#Generating my random data
#n is the number of rows in the columns in the matrix we want to create

function generate_real_data(n)
  Random.seed!(1)
  x1 = rand(1, n) .- 0.5
  x2 = (x1 .* x1 ) * 3 .+ rand(1, n) * 0.5 
  final = vcat(x1, x2)
  return final
end  

function generate_fake_data_1(n)
  Random.seed!(1)
  th  = 2*π*rand(1,n)
  r  = rand(1,n)/3
  x1 = @. r * cos(th)
  x2 = @. r*sin(th) + 0.5
  final = vcat(x1, x2)
return final 
end

function generate_fake_data_2(n)
  Random.seed!(1)
  th  = 2*π*rand(1,n) .+ 0.5
  r  = rand(1,n)/3 .+ 0.5
  x1 = @. r * cos(th)
  x2 = @. r*sin(th) + 1.5
  final = vcat(x1, x2)
return final 
end

train_size = 10000
real_data = generate_real_data(train_size)
fake_data_1 = generate_fake_data_1(train_size)
fake_data_2 = generate_fake_data_2(train_size);

scatter(real_data[1,1:train_size], real_data[2,1:train_size], label = "Real Data")

scatter(fake_data_1[1,1:train_size], fake_data_1[2,1:train_size], label = "Fake Data_1", color = "orange")
scatter!(fake_data_2[1,1:train_size], fake_data_2[2,1:train_size], label = "Fake Data_2", color = "red")

scatter(real_data[1,1:train_size], real_data[2,1:train_size], label = "Real Data")
scatter!(fake_data_1[1,1: train_size], fake_data_1[2,1:train_size], label = "Fake Data 1")
scatter!(fake_data_2[1,1: train_size], fake_data_2[2,1:train_size], label = "Fake Data 2")


#test one onehotencoding with my function 
X = hcat(real_data, fake_data_1, fake_data_2)


l0 = []
for i in 1:train_size
  append!(l0, 0)
end 

l2 = []
for i in 1:train_size
  append!(l2, 2)
end 

Y_ = vcat(ones(train_size),l0, l2)

Y = one_hot_encode_matrix(Y_, 3)


train_x, test_x = split_matrix(X, 0.66)
train_y, test_y = split_matrix(Y', 0.66)

#check the dimentions of training and testing dataset
println("size(X): ", size(X))
println("size(Y): ", size(Y'))
println("size(train_x): ", size(train_x))
println("size(train_y): ", size(train_y))
println("size(test_x): ", size(test_x))
println("size(test_y): ", size(test_y))





data = Flux.Data.DataLoader((X, Y'), batchsize=100,shuffle=true)
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

plot(epochs, loss_on_train, lab="Training loss", c=:black, lw=2);
plot!(epochs, loss_on_test, lab="Validation loss", c=:teal, ls=:dot);
yaxis!("Loss", :log);
xaxis!("Training epoch")

scatter(real_data[1,1:train_size],real_data[2,1:train_size],zcolor=m(real_data)')
scatter!(fake_data_1[1,1:train_size],fake_data_1[2,1:train_size],zcolor=m(fake_data_1)',legend=false)
scatter!(fake_data_2[1,1:train_size],fake_data_2[2,1:train_size],zcolor=m(fake_data_2)',legend=false)

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

