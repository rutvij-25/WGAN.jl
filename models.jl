using Flux

disc_block(in, out, k, s, p) = Chain(
    Conv((k,k),in=>out,stride = s,pad = p,bias=false),
    BatchNorm(out,affine=true),
    x -> leakyrelu.(x,0.2)
)

function Discriminator()
    return Chain(
        Conv((4,4),1=>64,stride=2,pad=1),
        x -> leakyrelu.(x,0.2),
        disc_block(64,128,4,2,1),
        disc_block(128,256,4,2,1),
        disc_block(256,512,4,2,1),
        Conv((4,4),512=>1,stride=2,pad=0)
    )
end

function discriminator_loss(real_output, fake_output)
    real_output = reshape(real_output,:)
    fake_output = reshape(fake_output,:)
    loss = -(mean(real_output) - mean(fake_output))
    return loss
end

function generator_loss(fake_output)
    fake_output = reshape(fake_output,:)
    return -mean(fake_output)
end