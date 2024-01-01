import Pkg; Pkg.add("Plots");
import Pkg; Pkg.add("StatsBase");
import Pkg; Pkg.add("InvertedIndices")
import Pkg; Pkg.add("Flux")
import Pkg; Pkg.add("OneHotArrays")
import Pkg; Pkg.add("Graphs")
import Pkg; Pkg.add("GraphPlot")
import Pkg; Pkg.add("Distributions")
Pkg.add("PyCall")

using Random
using InvertedIndices
using SparseArrays
using StatsBase
using LinearAlgebra
using Plots
using BenchmarkTools
using Flux
using PyCall
using OneHotArrays
using Statistics
using Graphs
using GraphPlot
using Distributions
np = pyimport("numpy")

#convert adj to S marix
function A2S(AdjMat)
    AdjMat += I #add the identity to the diagonal, add self-loops
    diag = Diagonal(vec(sum(AdjMat,dims=2) .^ (-1/2)))
    return (diag) * AdjMat * (diag) #return the normalized S matrix
end

#This function makes data using Normal or Uniform distribution
function create_adjacency_matrix_normal_uniform(num, distro1)
  Random.seed!(10)
  ER_tmp = erdos_renyi( num , 10*(num) )
  BA_tmp = barabasi_albert( num , 8 )
  SF_tmp = static_scale_free( num , 8*(num) , 4 )
  WS_tmp = erdos_renyi( num , 10*(num) ) #barabasi_albert( NN_tmp , 5 )
  blocks_tmp = blockdiag( blockdiag( blockdiag(ER_tmp,BA_tmp),SF_tmp ), WS_tmp )
    
  #now add some edges between the blocks that are the communities
  for bb in 1:Int(round(num/10))
    for b1 in 0:3
      for b2 in 0:3
        if(b1 < b2)
          range1 = randperm(num)[1] + b1*num
          range2 = randperm(num)[1] + b2*num
          add_edge!( blocks_tmp , range1 , range2 )
        end
      end
    end
  end
  density_nn = Graphs.density(blocks_tmp)
  adj = Matrix(adjacency_matrix(blocks_tmp))

 
  d1 = rand(distro1(0, 1), 3 * num)
  d2 = rand(distro1(0, 1), 3 * num)
  d3 = rand(distro1(0, 1), 3 * num)
  c1 = Categorical( [0.5,0.25,0.25] )
  c2 = Categorical( [0.15,0.15,0.7] )
  c3 = Categorical( [0.5,0.5,0] )

 

   xd1 = rand( d1 , 3 * num )
   xd1 = reshape(xd1, 3 ,num)
   xc1 = onehotbatch( rand( c1 , num ) , 1:3 )
   x1a = vcat( xd1 , xc1 )'
   xd1 = rand( d1 , 3 * num )
   xd1 = reshape(xd1, 3 ,num)
   xc1 = onehotbatch( rand( c1 , num ) , 1:3 )
   x1b = vcat( xd1 , xc1 )'
   xd2 = rand( d2 ,  3 * num )
   xd2 = reshape(xd2, 3 ,num)
   xc2 = onehotbatch( rand( c2 , num ) , 1:3 )
   x2 = vcat( xd2 , xc2 )'
   xd3 = rand( d3 ,  3 * num )
   xd3 = reshape(xd3, 3 ,num)
   xc3 = onehotbatch( rand( c3 , num ) , 1:3 )
   x3 = vcat( xd3 , xc3 )'
   xc3 = onehotbatch( rand( c3 , num ) , 1:3 )
   x3 = vcat( xd3 , xc3 )'

  X = vcat( x1a , x1b , x2 , x3 )
  y1a = onehotbatch( 1*ones(num) , 1:2 )'
  y1b = onehotbatch( 1*ones(num) , 1:2 )'
  y2 = onehotbatch( 2*ones(num) , 1:2 )'
  y3 = onehotbatch( 2*ones(num) , 1:2 )'
  Y = vcat(y1a, y1b, y2, y3)
  Y_to_use = vcat(1*ones(num), 1*ones(num), 2*ones(num), 2*ones(num))
  #println("size(X): ", size(X))
  #println("size(Y):", size(Y_to_use))
  return adj, X, Y_to_use
end 

#this function checks if I the y_hat data predicted by the network are the same with the y_hat_manually data, which is the data I predicted with the classical way: F(WX + bias)
#vec1 = y_hat
#vec2 = y_hat_manually

function check_validity(vec1, vec2)
  s = 0
  for i in 1:length(vec1)
    if vec1[i] == vec2[i]
      s += 1
    end
  end
  if s == length(vec1)
    println("The two methods match")
  else
    println("Methods do not match")
  end 
end 

#This function loads the data and trains the neural network. 
#SX_ = the S*X matrix (X data)
#yhot = the Y data with (onehotbatch)

function load_and_train_model(SX_, yhot_)
  resDict_ = Dict()
  model_ = Chain( Dense( size(SX_, 1) => size(yhot_, 1)) , softmax)
  loss(x, y) = Flux.crossentropy(model_(x), y)
  opt = Adam(0.01)
  pars = Flux.params(model_)
  data = Flux.DataLoader((SX_, yhot_) , batchsize = 10 , shuffle = true)
  epochs_ = Int64[]
  for epoch in 1:500
    Flux.train!(loss, pars, data ,opt)
    push!(epochs_, epoch)
  end 
  resDict_["params"] = pars
  resDict_["model"] = model_
  return(model_, resDict_)
end 



num = 25
method = 0
distro1 = Normal 
for k in 1
    ad, x, y= create_adjacency_matrix_normal_uniform(num, distro1)
    yhot = onehotbatch(y, [1, 2])
    S = A2S(ad)
    SX = S^k * x
    SX
    SX = SX'
    println("SX = ", SX)
    train_x = SX
    println("train_x = ", train_x)
    train_y = yhot
    if method == 0
      #println("Training with raw data")
      #epochs, loss_on_train, loss_on_test,  model, resDict = load_and_train_3(SX, yhot, train_x, train_y)
      model, resDict = load_and_train_model(SX, yhot)
      weight = resDict["model"].layers[1].weight
      println("weight = ", weight)
      accuracy = round(mean( onecold( model(train_x), [1, 2] ) .== onecold(train_y, [1, 2]) ) * 100, digits = 2)
      println("Accuracy: ", accuracy, "%", ", k = $k, distribution = $distro1")      
      y_hat = onecold( model(train_x), [1, 2])
      y_actual = onecold(train_y, [1, 2])
      println("y_hat = ", y_hat)
      test = weight * SX
      bias_ = resDict["model"].layers[1].bias
      bias_1 = fill(bias_[1], (1, size(test, 2)))
      bias_2 = fill(bias_[2], (1, size(test, 2)))
      bias = vcat(bias_1, bias_2)
      z = test + bias
      z_pred = softmax(z)
      y_hat_manually = onecold(softmax(z), [1, 2])
      println("y_hat_manually = ", y_hat_manually)
      check_validity(y_hat, y_hat_manually)
    end 
  end


