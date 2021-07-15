using Flux
using Zygote

function gradient_penalty(dscr,real,fake)

    H, W, C, B = size(real)
    alpha = rand(1,1,1,B)
    alpha = repeat(alpha,H,W,C,1)
    interpolated_image = real .* alpha + fake .* (1 .- alpha)
    gp = gradient(params[interpolated_image]) do 
        
        
    end
    
    return gp[interpolated_image]
end