#using Plots
using Distributions
using StatsBase
using InvertedIndices
using OneHotArrays
using Graphs
using GraphPlot
using Distributions
using LinearAlgebra
using Random 
using InvertedIndices

############################################## generate data to test the my_function ##############################################


#convert adj to S marix
  function A2S(AdjMat)
    AdjMat += I #add the identity to the diagonal, add self-loops
    diag = Diagonal(vec(sum(AdjMat,dims=2) .^ (-1/2)))
    return (diag) * AdjMat * (diag) #return the normalized S matrix
end

#This function generates the Dataset

function AdjMat_XMat_YMat()
  Random.seed!(10)
  nn_max = 3
  NN_set = 10^nn_max
  NN_tmp = Int( NN_set / 4 )
  ER_tmp = erdos_renyi( NN_tmp , 10*(NN_tmp) )
  BA_tmp = barabasi_albert( NN_tmp , 8 )
  SF_tmp = static_scale_free( NN_tmp , 8*(NN_tmp) , 4 )
  WS_tmp = erdos_renyi( NN_tmp , 10*(NN_tmp) ) #barabasi_albert( NN_tmp , 5 )
  blocks_tmp = blockdiag( blockdiag( blockdiag(ER_tmp,BA_tmp),SF_tmp ), WS_tmp )
    
  ER_BA_SF_WS_Block_graphs = blockdiag( blockdiag( blockdiag(ER_tmp,BA_tmp),SF_tmp ), WS_tmp )
  #now add some edges between the blocks that are the communities
  for bb in 1:Int(round(NN_tmp/10))
    for b1 in 0:3
      for b2 in 0:3
        if(b1 < b2)
          range1 = randperm(NN_tmp)[1] + b1*NN_tmp
          range2 = randperm(NN_tmp)[1] + b2*NN_tmp
          add_edge!( ER_BA_SF_WS_Block_graphs , range1 , range2 )
        end
      end
    end
  end 
  #density_nn = Graphs.density(ER_BA_SF_WS_Block_graphs)
  ER_BA_SF_WS_Block_matrices = Matrix(adjacency_matrix(ER_BA_SF_WS_Block_graphs))

  d1 = Dirichlet( [10,10,10] )
  c1 = Categorical( [0.5,0.25,0.25] )
  d2 = Dirichlet( [20,10,10] )
  c2 = Categorical( [0.35,0.35,0.3] )
  d3 = Dirichlet( [20,10,20] )
  c3 = Categorical( [0.25,0.25,0.5] )

  networks_X = Dict()
  networks_Y = Dict()
  networks_Y_cold = Dict()

    
    NN_tmp = Int( NN_set / 4 )
    
    xd1 = rand( d1 , NN_tmp )
    xc1 = onehotbatch( rand( c1 , NN_tmp ) , 1:3 )
    x1a = vcat( xd1 , xc1 )'
    xd1 = rand( d1 , NN_tmp )
    xc1 = onehotbatch( rand( c1 , NN_tmp ) , 1:3 )
    x1b = vcat( xd1 , xc1 )'
    xd2 = rand( d2 , NN_tmp )
    xc2 = onehotbatch( rand( c2 , NN_tmp ) , 1:3 )
    x2 = vcat( xd2 , xc2 )'
    xd3 = rand( d3 , NN_tmp )
    xc3 = onehotbatch( rand( c3 , NN_tmp ) , 1:3 )
    x3 = vcat( xd3 , xc3 )'
    xc3 = onehotbatch( rand( c3 , NN_tmp ) , 1:3 )
    x3 = vcat( xd3 , xc3 )'

    networks_X = vcat( x1a , x1b , x2 , x3 )
    XMat = networks_X
    
    y1a = onehotbatch( 1*ones(NN_tmp) , 1:2 )'
    y1b = onehotbatch( 1*ones(NN_tmp) , 1:2 )'
    y2 = onehotbatch( 2*ones(NN_tmp) , 1:2 )'
    y3 = onehotbatch( 2*ones(NN_tmp) , 1:2 )'
    
    networks_Y = vcat( y1a , y1b , y2 , y3 )
    networks_Y_cold = vcat(1*ones(NN_tmp),1*ones(NN_tmp),2*ones(NN_tmp),2*ones(NN_tmp))       
    YMat = networks_Y_cold
    return ER_BA_SF_WS_Block_matrices, XMat, YMat
end 

 adj_matrix, x_matrix, y_matrix  = AdjMat_XMat_YMat();
