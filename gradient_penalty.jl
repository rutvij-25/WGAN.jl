using Flux
using Zygote:jacobian

function gradient_penalty(dscr,real,fake)

    H, W, C, B = size(real)
    alpha = rand(1,1,1,B)
    alpha = repeat(alpha,H,W,C,1)
    interpolated_image = real .* alpha + fake .* (1 .- alpha)
    gp = jacobian(()->dscr(interpolated_image),params(interpolated_image))
    
    return gp[interpolated_image]
end