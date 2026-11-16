# TODO Improve load balance when iterating over combination of Section/SectionDiffs and Points.
# TODO Switch to Zhichao's new data.
# TODO Finish MLE and check for stability.
# TODO Target estimates:
#      - Time series of Scholte wave travel time
#      - Trend section along Scholte wave path
#      - Trend profile averaged over Scholte wave path

using HDF5, CSV, DataFrames, Dates, TriangularIndices, LinearAlgebra, Base.Threads, FileIO, NCDatasets, Interpolations, Statistics, GibbsSeaWater, Printf, ProgressMeter, PyPlot, Random, TiffImages

"""
    meanyear
Mean length of year, taking leap years into account.
"""
global const meanyear = 365.2425

global const meanmonth = Second(2629746) # meanyear/12

"""
    reftime
Reference time used for conversion to float time.

See also [`dt2t`](@ref), [`t2dt`](@ref).
"""
global const reftime = DateTime("2008-08-30T12:00:00")

"""
    dt2t(dt)
Convert the `DateTime` object `dt` to a float time (in years) referenced to `reftime`.

See also [`reftime`](@ref), [`t2dt`](@ref).
"""
dt2t(dt) = Dates.value(dt-reftime)/1e3/86400/meanyear

"""
    t2dt(t)
Convert the float time `t`, referenced to `reftime`, to a `DateTime` object.

See also [`reftime`](@ref), [`dt2t`](@ref).
"""
t2dt(t) = reftime + Millisecond(round(t*1e3*86400*meanyear))

"""
    earthradius
Radius of the reference sphere.
"""
global const earthradius = 6371.0

"""
    catesian(lat, lon)
Convert from latitude and longitude to Cartesian coordinates on the reference sphere.
"""
cartesian(lat, lon) = earthradius*[cosd(lat)*cosd(lon), cosd(lat)*sind(lon), sind(lat)]

"""
    spherical(x, y, z)
Convert from Cartesian coordinates on the reference sphere to latitude and longitude.
"""
spherical(x, y, z) = [asind(z/hypot(x, y, z)), atand(y, x)]

"""
    HorizontalPosition
Horizontal position on the reference sphere, with an implicit conversion to Cartesian coordinates.
"""
struct HorizontalPosition
    lat::Float64
    lon::Float64
    cartpos::Vector{Float64}
    @doc """
        HorizontalPosition(lat, lon)
    Construct a `HorizontalPosition` with location `lat`, `lon` on the reference sphere. This automatically generates a `cartpos` field that holds the corresponding Cartesian position.
    """
    HorizontalPosition(lat::Float64, lon::Float64) = new(lat, lon, cartesian(lat, lon))
end

Base.show(io::IO, ::MIME"text/plain", hpos::HorizontalPosition) = print(io, "HorizontalPosition:\n├─ lat: ", hpos.lat, "\n└─ lon: ", hpos.lon)
Base.show(io::IO, hpos::HorizontalPosition) = print(io, "(", hpos.lat, ", ", hpos.lon, ")")

"""
    Kernel
Sensitivity kernel along a depth–range section.
"""
struct Kernel
    hpos::Vector{HorizontalPosition}
    depth::Vector{Float64}
    weight::Matrix{Float64} # sensitivity kernel × Δr × Δz
    nr::Int64
    nz::Int64
    function Kernel(hpos, depth, weight)
        nr = length(hpos)
        nz = length(depth)
        (nr, nz) != size(weight) && error("size of weight array not consistent with positions")
        new(hpos, depth, weight, nr, nz)
    end
end

"""
    Kernel(lat, lon, depth, weight)
Construct a `Kernel` along the section specified by `lat`, `lon`, and `depth`, with the weight at each point given by `weight`. These weights are intepreted as the sensitivity kernel integrated over the corresponding area element.
"""
Kernel(lat::Vector{Float64}, lon::Vector{Float64}, depth::Vector{Float64}, weight::Matrix{Float64}) = Kernel(HorizontalPosition.(lat, lon), depth, weight)

Base.size(k::Kernel) = (k.nr, k.nz)
Base.broadcastable(k::Kernel) = Ref(k)

Base.show(io::IO, ::MIME"text/plain", k::Kernel) = print(io, k.nr, "×", k.nz, " Kernel:\n├─ lat: ", k.hpos[1].lat, " … ", k.hpos[end].lat, "\n├─ lon: ", k.hpos[1].lon, " … ", k.hpos[end].lon, "\n├─ depth: ", k.depth[1], " … ", k.depth[end], "\n└─ weight: sum = ", sum(k.weight), ", min = ", minimum(k.weight), ", max = ", maximum(k.weight))
Base.show(io::IO, k::Kernel) = print(io, "Kernel(", k.nr, ", ", k.nz, ")")

"Representation of a measurement (hydrographic or seismic)."
abstract type Representation end

"""
    Point <: Representation
Representation of a point measurement.
"""
struct Point <: Representation
    hpos::HorizontalPosition
    depth::Float64
    time::Float64
end

"""
    Point(lat, lon, depth, time)
Construct a `Point` object with horizontal position `lat`, `lon`, at `depth`, and at `time` (given as `DateTime` object).
"""
Point(lat::Float64, lon::Float64, depth::Float64, time::DateTime) = Point(HorizontalPosition(lat, lon), depth, dt2t(time))

Base.show(io::IO, ::MIME"text/plain", p::Point) = print(io, "Point:\n├─ lat: ", p.hpos.lat, "\n├─ lon: ", p.hpos.lon, "\n├─ depth: ", p.depth, "\n└─ time: ", t2dt(p.time))
Base.show(io::IO, p::Point) = print(io, "Point(", p.hpos.lat, ", ", p.hpos.lon, ", ", p.depth, ", ", t2dt(p.time), ")")

"""
    Section <: Representation
Representation of a seismic travel time anomaly, interpreted as a weighted average over a depth–range section.
"""
struct Section <: Representation
    hpos::Vector{HorizontalPosition}
    depth::Vector{Float64}
    time::Float64
    weight::Matrix{Float64} # sensitivity kernel × Δr × Δz
    nr::Int64
    nz::Int64
    function Section(hpos, depth, time, weight)
        nr = length(hpos)
        nz = length(depth)
        (nr, nz) != size(weight) && error("size of weight array not consistent with positions")
        new(hpos, depth, time, weight, nr, nz)
    end
end

"""
    Section(kernel, time)
Construct a `Section` object along the section and with weights as specified by the `kernel`, at `time` (given as `DateTime` object).
"""
Section(k::Kernel, time::DateTime) = Section(k.hpos, k.depth, dt2t(time), k.weight)

"""
    Section(lat, lon, depth, time, weight)
Construct a `Section` object along the horizontal locations `lat`, `lon`, at `depth`, and at `time` (given as `DateTime` object). The weights along the section are specified by `weight`, which are the sensitivity kernels multiplied by a corresponding area element.
"""
Section(lat::Vector{Float64}, lon::Vector{Float64}, depth::Vector{Float64}, time::DateTime, weight::Matrix{Float64}) = Section(HorizontalPosition.(lat, lon), depth, dt2t(time), weight)

Base.size(s::Section) = (s.nr, s.nz)
Base.broadcastable(s::Section) = Ref(s)

Base.show(io::IO, ::MIME"text/plain", s::Section) = print(io, s.nr, "×", s.nz, " Section:\n├─ lat: ", s.hpos[1].lat, " … ", s.hpos[end].lat, "\n├─ lon: ", s.hpos[1].lon, " … ", s.hpos[end].lon, "\n├─ depth: ", s.depth[1], " … ", s.depth[end], "\n├─ time: ", t2dt(s.time), "\n└─ weight: sum = ", sum(s.weight), ", min = ", minimum(s.weight), ", max = ", maximum(s.weight))
Base.show(io::IO, s::Section) = print(io, "Section(", s.nr, ", ", s.nz, ")")

"""
    SectionDiff <: Representation
Representation of a seismic travel time change measurement, interpreted as a weighted average over a depth–range section.
"""
struct SectionDiff <: Representation
    hpos::Vector{HorizontalPosition}
    depth::Vector{Float64}
    time1::Float64 # first event
    time2::Float64 # second event
    weight::Matrix{Float64} # sensitivity kernel × Δr × Δz
    nr::Int64
    nz::Int64
    function SectionDiff(hpos, depth, time1, time2, weight)
        nr = length(hpos)
        nz = length(depth)
        (nr, nz) != size(weight) && error("size of weight array not consistent with positions")
        new(hpos, depth, time1, time2, weight, nr, nz)
    end
end

"""
    SectionDiff(kernel, time1, time2)
Construct a `SectionDiff` object along the section and with weights as specified by the `kernel`. The first and second event times are specified by `time1` and `time2`, respectively (given as `DateTime` objects).
"""
SectionDiff(k::Kernel, time1::DateTime, time2::DateTime) = SectionDiff(k.hpos, k.depth, dt2t(time1), dt2t(time2), k.weight)

"""
    SectionDiff(lat, lon, depth, time1, time2, weight)
Construct a `SectionDiff` object along the horizontal locations `lat`, `lon`, at the depths `depth`, and with the first and second event times `time1` and `time2`, respectively (given as `DateTime` objects). The weights along the section are specified by `weight`, which are the sensitivity kernels multiplied by a corresponding area element.
"""
SectionDiff(lat::Vector{Float64}, lon::Vector{Float64}, depth::Vector{Float64}, time1::DateTime, time2::DateTime, weight::Matrix{Float64}) = SectionDiff(HorizontalPosition.(lat, lon), depth, dt2t(time1), dt2t(time2), weight)

Base.size(s::SectionDiff) = (s.nr, s.nz)
Base.broadcastable(s::SectionDiff) = Ref(s)

Base.show(io::IO, ::MIME"text/plain", s::SectionDiff) = print(io, s.nr, "×", s.nz, " SectionDiff:\n├─ lat: ", s.hpos[1].lat, " … ", s.hpos[end].lat, "\n├─ lon: ", s.hpos[1].lon, " … ", s.hpos[end].lon, "\n├─ depth: ", s.depth[1], " … ", s.depth[end], "\n├─ times: ", t2dt(s.time1), ", ", t2dt(s.time2), "\n└─ weight: sum = ", sum(s.weight), ", min = ", minimum(s.weight), ", max = ", maximum(s.weight))
Base.show(io::IO, s::SectionDiff) = print(io, "SectionDiff(", s.nr, ", ", s.nz, ")")

"""
    hdist(p, q)

Calculate the cordal distance between the horizontal positions `p` and `q` on the reference sphere.
"""
hdist(p::HorizontalPosition, q::HorizontalPosition) = norm(p.cartpos - q.cartpos)

global const nparams = 5
global const ndepths = 12

# map from depth to index
depthidx(depth) = Int(round((depth+250.0)/500.0))

# covariance of reference profile
covref(θ, i, j) = θ[nparams+i]*θ[nparams+j]

# covariance of time mean deviation from reference
covmean(θ, i, j, d, dz) = θ[nparams+ndepths+i]*θ[nparams+ndepths+j]*exp(-d/θ[2] - dz/θ[1])

# covariance of the seasonal cycle
covseason(θ, i, j, d, dt) = θ[nparams+2ndepths+i]*θ[nparams+2ndepths+j]*exp(-d/θ[2])*cos(2π*dt)

# covariance of the trend
covtrend(θ, i, j, d, dz, time1, time2) = θ[nparams+3ndepths+i]*θ[nparams+3ndepths+j]*exp(-d/θ[2] - dz/θ[1])*time1*time2

# covariance of eddy anomalies
coveddy(θ, i, j, d, dz, dt) = θ[nparams+4ndepths+i]*θ[nparams+4ndepths+j]*exp(-d/θ[3] - dz/θ[4] - dt/θ[5])

function covall(θ::Vector{Float64}, d::Float64, depth1::Float64, depth2::Float64, time1::Float64, time2::Float64)
    # vertical distance
    dz = abs(depth1 - depth2)
    # time difference
    dt = abs(time1 - time2)
    # find depth indices
    i = depthidx(depth1)
    j = depthidx(depth2)
    return covref(θ, i, j) + covmean(θ, i, j, d, dz) + covseason(θ, i, j, d, dt) + covtrend(θ, i, j, d, dz, time1, time2) + coveddy(θ, i, j, d, dz, dt)
end

function covref(θ::Vector{Float64}, d::Float64, depth1::Float64, depth2::Float64, time1::Float64, time2::Float64)
    # find depth indices
    i = depthidx(depth1)
    j = depthidx(depth2)
    return covref(θ, i, j)
end

function covmean(θ::Vector{Float64}, d::Float64, depth1::Float64, depth2::Float64, time1::Float64, time2::Float64)
    # vertical distance
    dz = abs(depth1 - depth2)
    # find depth indices
    i = depthidx(depth1)
    j = depthidx(depth2)
    return covmean(θ, i, j, d, dz)
end

function covtrend(θ::Vector{Float64}, d::Float64, depth1::Float64, depth2::Float64, time1::Float64, time2::Float64)
    # vertical distance
    dz = abs(depth1 - depth2)
    # find depth indices
    i = depthidx(depth1)
    j = depthidx(depth2)
    return covtrend(θ, i, j, d, dz, time1, time2)
end

function covanom(θ::Vector{Float64}, d::Float64, depth1::Float64, depth2::Float64, time1::Float64, time2::Float64)
    # vertical distance
    dz = abs(depth1 - depth2)
    # time difference
    dt = abs(time1 - time2)
    # find depth indices
    i = depthidx(depth1)
    j = depthidx(depth2)
    return covseason(θ, i, j, d, dt) + covtrend(θ, i, j, d, dz, time1, time2) + coveddy(θ, i, j, d, dz, dt)
end

function cov(c::Function, θ::Vector{Float64}, s::SectionDiff, t::SectionDiff)
    C = 0.0
    for i = 1:s.nr, j = 1:t.nr
        d = hdist(s.hpos[i], t.hpos[j])
        for k = 1:s.nz, l = 1:t.nz
            C += s.weight[i,k]*t.weight[j,l]*c(θ, d, s.depth[k], t.depth[l], s.time1, t.time1)
            C -= s.weight[i,k]*t.weight[j,l]*c(θ, d, s.depth[k], t.depth[l], s.time1, t.time2)
            C -= s.weight[i,k]*t.weight[j,l]*c(θ, d, s.depth[k], t.depth[l], s.time2, t.time1)
            C += s.weight[i,k]*t.weight[j,l]*c(θ, d, s.depth[k], t.depth[l], s.time2, t.time2)
        end
    end
    return C
end

function cov(c::Function, θ::Vector{Float64}, s::SectionDiff, p::Point)
    C = 0.0
    for i = 1:s.nr
        d = hdist(s.hpos[i], p.hpos)
        for j = 1:s.nz
            C -= s.weight[i,j]*c(θ, d, s.depth[j], p.depth, s.time1, p.time)
            C += s.weight[i,j]*c(θ, d, s.depth[j], p.depth, s.time2, p.time)
        end
    end
    return C
end

cov(c::Function, θ::Vector{Float64}, p::Point, s::SectionDiff) = cov(c, θ, s, p)

function cov(c::Function, θ::Vector{Float64}, s::Section, t::Section)
    C = 0.0
    for i = 1:s.nr, j = 1:t.nr
        d = hdist(s.hpos[i], t.hpos[j])
        for k = 1:s.nz, l = 1:t.nz
            C += s.weight[i,k]*t.weight[j,l]*c(θ, d, s.depth[k], t.depth[l], s.time, t.time)
        end
    end
    return C
end

function cov(c::Function, θ::Vector{Float64}, s::Section, t::SectionDiff)
    C = 0.0
    for i = 1:s.nr, j = 1:t.nr
        d = hdist(s.hpos[i], t.hpos[j])
        for k = 1:s.nz, l = 1:t.nz
            C -= s.weight[i,k]*t.weight[j,l]*c(θ, d, s.depth[k], t.depth[l], s.time, t.time1)
            C += s.weight[i,k]*t.weight[j,l]*c(θ, d, s.depth[k], t.depth[l], s.time, t.time2)
        end
    end
    return C
end

function cov(c::Function, θ::Vector{Float64}, s::Section, p::Point)
    C = 0.0
    for i = 1:s.nr
        d = hdist(s.hpos[i], p.hpos)
        for j = 1:s.nz
            C += s.weight[i,j]*c(θ, d, s.depth[j], p.depth, s.time, p.time)
        end
    end
    return C
end

cov(c::Function, θ::Vector{Float64}, p::Point, s::Section) = cov(c, θ, s, p)

function cov(c::Function, θ::Vector{Float64}, p::Point, q::Point)
    d = hdist(p.hpos, q.hpos)
    return c(θ, d, p.depth, q.depth, p.time, q.time)
end

cov(c::Function, θ::Vector{Float64}, a::Nothing, r::Representation) = 0.0
cov(c::Function, θ::Vector{Float64}, r::Representation, a::Nothing) = 0.0
cov(c::Function, θ::Vector{Float64}, a::Nothing, b::Nothing) = 0.0

function cov(c::Function, θ::Vector{Float64}, r::Vector{T}) where {T<:Union{Representation, Nothing}}
    n = length(r)
    C = zeros(n, n)
    @showprogress @threads for (i, j) in shuffle(collect(UpperTriangularIndices(C)))
        C[i,j] = cov(c, θ, r[i], r[j])
    end
    # @sync @showprogress for (i, j) in UpperTriangularIndices(C)
    #     if typeof(r[i]) in (Section, SectionDiff) && typeof(r[j]) in (Section, SectionDiff)
    #         @spawn C[i,j] = cov(θ, r[i], r[j])
    #     else
    #         C[i,j] = cov(θ, r[i], r[j])
    #     end
    # end
    return Symmetric(C)
end

function cov(c::Function, θ::Vector{Float64}, r::Vector{T}, s::Vector{U}) where {T<:Union{Representation, Nothing}, U<:Union{Representation, Nothing}}
    n = length(r)
    m = length(s)
    C = zeros(n, m)
    @showprogress @threads for i = shuffle(CartesianIndices((n, m)))
        C[i] = cov(c, θ, r[i[1]], s[i[2]])
    end
    # @sync @showprogress for i = CartesianIndices((n, m))
    #     if typeof(r[i[1]]) in (Section, SectionDiff) && typeof(s[i[2]]) in (Section, SectionDiff)
    #         @spawn C[i] = cov(θ, r[i[1]], s[i[2]])
    #     else
    #         C[i] = cov(θ, r[i[1]], s[i[2]])
    #     end
    # end
    return C
end

function estimate(c::Function, θ::Vector{Float64}, r::Vector{T}, s::Vector{U}, y::Vector{Float64}, Ryy::Matrix{Float64}) where {T<:Representation, U<:Union{Representation, Nothing}}
    # prior covariance at estimation points
    Rxx = cov(c, θ, r)
    # covariance between estimation points and measurements
    Rxy = cov(c, θ, r, s)
    # solve
    F = cholesky(Ryy)
    x = Rxy*(F\y)
    v = F.L\Rxy'
    P = Rxx - v'*v
    return x, Rxx, P
end

function dcdT(lat_kernel, lon_kernel, depth_kernel)
    @assert(length(lat_kernel) == length(lon_kernel))
    # load WOA
    fl = NCDataset("../woadata/woa23_decav91C0_t00_01.nc")
    lat = fl["lat"][:]
    lon = fl["lon"][:]
    depth = fl["depth"][:]
    T = fl["t_an"][:,:,:,1]
    fl = NCDataset("../woadata/woa23_decav91C0_s00_01.nc")
    S = fl["s_an"][:,:,:,1]
    # shift such that dateline is in the middle of the domain
    lon_kernel = mod.(lon_kernel, 360)
    lon = mod.(circshift(lon, 180), 360)
    T = circshift(T, (180, 0, 0))
    S = circshift(T, (180, 0, 0))
    # interpolate in horizontal
    nz = length(depth)
    nr = length(lat_kernel)
    Tp = Array{Union{Missing, Float64}}(undef, nr, nz)
    Sp = Array{Union{Missing, Float64}}(undef, nr, nz)
    for i = 1:nz
        itp = interpolate((lon, lat,), T[:,:,i], Gridded(Linear()))
        Tp[:,i] = itp.(lon_kernel, lat_kernel)
        itp = interpolate((lon, lat,), S[:,:,i], Gridded(Linear()))
        Sp[:,i] = itp.(lon_kernel, lat_kernel)
    end
    # interpolate in vertical
    nz = length(depth_kernel)
    Tk = Array{Union{Missing, Float64}}(undef, nr, nz)
    Sk = Array{Union{Missing, Float64}}(undef, nr, nz)
    for i = 1:nr
        itp = extrapolate(interpolate((depth,), Tp[i,:], Gridded(Linear())), Flat())
        Tk[i,:] = itp.(depth_kernel)
        itp = extrapolate(interpolate((depth,), Sp[i,:], Gridded(Linear())), Flat())
        Sk[i,:] = itp.(depth_kernel)
    end
    # fill in missing values
    Tm = [mean(skipmissing(Tk[:,i])) for i = 1:nz]
    Sm = [mean(skipmissing(Sk[:,i])) for i = 1:nz]
    for i = 1:nz
        Tk[ismissing.(Tk[:,i]),i] .= Tm[i:-1:1][findfirst(.!isnan.(Tm[i:-1:1]))]
        Sk[ismissing.(Sk[:,i]),i] .= Sm[i:-1:1][findfirst(.!isnan.(Sm[i:-1:1]))]
    end
    # convert to pressure, conservative temperature, absolute salinity
    p = gsw_p_from_z.(-depth_kernel', lat_kernel)
    SA = gsw_sa_from_sp.(Sk, p, lon_kernel, lat_kernel)
    CT = gsw_ct_from_t.(SA, Tk, p)
    δ = 1e-3
    α = (gsw_sound_speed.(SA, CT.+δ, p) - gsw_sound_speed.(SA, CT.-δ, p))/2δ
    return α
end

"""
    loadkernels(path, num, freq, Δr, Δz)
Load `num` sensitivity kernels from files at `path`, grabbing the lowest `freq` frequencies. The kernels are subsampled in the vertical and integrated over area elements according to the range spacing `Δr` and the depth spacing `Δz`.
"""
function loadkernels(path::String, num::Int, freq::Int, Δr::Float64, Δz::Float64)
    kernels = Array{Kernel}(undef, num, freq)
    @showprogress for i = 1:num, j = 1:freq
        # read from files
        filename = path * @sprintf("%04d.h5", i-1)
        K = Float64.(h5read(filename, "kernels_SEM_masked")[j,:,:])
        lat = Float64.(h5read(filename, "lat_grid")[:,1])
        lon = Float64.(h5read(filename, "lon_grid")[:,1])
        depth = Float64.(1f3h5read(filename, "depth_grid")[1,:])
        # zero sensitivity where there's no ocean
        replace!(K, NaN=>0.0)
        # convert to temperature kernel
        K .*= dcdT(lat, lon, depth)
        # include zeros at surface
        nr, nz = size(K)
        depth = [0; depth]
        K = [zeros(nr) K]
        # downsample and multiply by area element
        depth = [(depth[j]/2 + sum(depth[j+1:j+9]) + depth[j+10]/2)/10 for j = 1:10:120]
        K = [(K[i,j]/2 + sum(K[i,j+1:j+9]) + K[i,j+10]/2)*Δz for i = 1:nr, j = 1:10:120] # to 500 m depth intervals
        K = [sum(K[i:i+2,j])*Δr for i = 1:5:nr-4, j = 1:12] # to 25 km range intervals
        lat = lat[3:5:nr-2]
        lon = lon[3:5:nr-2]
        # save Kernel struct
        kernels[i,j] = Kernel(lat, lon, depth, K)
    end
    return kernels
end

"""
    loadtwavedata(pairs, kernels, freq)
Load T wave measurements from the list `pairs` and representing them with the `kernels` indicated in the list. This selects measurements at the `freq` lowest frequencies.
"""
function loadtwavedata(pairs, kernels, freq)
    n = size(pairs, 1)
    # read T-wave Δτ and its error
    y = Vector{Float64}(vcat([parse.(Float64, split(chop(pairs.TΔτc[i]; head=1, tail=1), ','))[1:freq] for i = 1:n]...))
    σ = Vector{Float64}(vcat([parse.(Float64, split(chop(pairs.TΔτc_error[i]; head=1, tail=1), ','))[1:freq] for i = 1:n]...))
    # store section information
    s = Vector{SectionDiff}(undef, n*freq)
    for i = 1:n, j = 1:freq
        k = kernels[parse(Int, pairs.Kernel_ID[i][15:18]) + 1,j]
        s[(j-1)*n+i] = SectionDiff(k, pairs.EvtA_time[i], pairs.EvtB_time[i])
    end
    return y, σ, s
end

"""
    loadscholtedata(pairs, kernels, freq)
Load Scholte wave measurements from the list `pairs` and representing them with the `kernels` indicated in the list. This selects measurements at the `freq` lowest frequencies.
"""
function loadscholtedata(pairs, kernels, freq)
    n = size(pairs, 1)
    # read T-wave Δτ and its error
    if freq == 1
        y = pairs.ScΔτ_75mHz
        σ = pairs.ScΔτ_75mHz_error
    elseif freq == 2
        y = [pairs.ScΔτ_75mHz; pairs.ScΔτ_85mHz]
        σ = [pairs.ScΔτ_75mHz_error; pairs.ScΔτ_85mHz_error]
    else
        error("We only have Scholte wave measurementes at two frequencies; `freq` must be `1` or `2`.")
    end
    # store section information
    s = Vector{SectionDiff}(undef, n*freq)
    for i = 1:n, j = 1:freq
        k = kernels[parse(Int, pairs.Kernel_ID[i][17:20]) + 1,j]
        s[(j-1)*n+i] = SectionDiff(k, pairs.EvtA_time[i], pairs.EvtB_time[i])
    end
    return y, σ, s
end

"""
    loadpwavedata(tpairs, spairs, ppairs, e)
Load the P-wave measurements contained in `ppairs` and associate them with the events in `tpairs` and `spairs`, then set up the origin time covariance matrix with the specified prior origin time uncertainty `e`.
"""
function loadpwavedata(tpairs, spairs, ppairs, tfreq, sfreq, e)
    # unique events
    events = sort(unique([ppairs.EvtA_time; ppairs.EvtB_time]))
    # lists of pairs
    tidx = Tuple{Int, Int}[]
    sidx = Tuple{Int, Int}[]
    pidx = Tuple{Int, Int}[]
    # P-wave measurements
    y = Float64[]
    σ = Float64[]
    # loop over T-wave pairs
    for p = eachrow(tpairs)
        # T-wave pair
        i = findfirst(events.==p.EvtA_time)
        j = findfirst(events.==p.EvtB_time)
        push!(tidx, (i, j))
        # find associated P-wave pair
        idx = findfirst(p.PairID.==ppairs.PairID)
        q = ppairs[idx,:]
        # P-wave measurements
        yf = parse.(Float64, split(chop(q.PΔτ; head=1, tail=1), ','))
        σf = parse.(Float64, split(chop(q.PΔτ_error; head=1, tail=1), ','))
        # save all available
        for k = 1:length(yf)
            if !isnan(yf[k])
                push!(pidx, (i, j))
                push!(y, yf[k])
                push!(σ, σf[k])
            end
        end
    end
    # loop over Scholte pairs
    for p = eachrow(spairs)
        # Scholte wave pair
        i = findfirst(events.==p.EvtA_time)
        j = findfirst(events.==p.EvtB_time)
        push!(sidx, (i, j))
        # find associated P-wave pair
        idx = findfirst(p.PairID.==ppairs.PairID)
        q = ppairs[idx,:]
        # P-wave measurements
        yf = parse.(Float64, split(chop(q.PΔτ; head=1, tail=1), ','))
        σf = parse.(Float64, split(chop(q.PΔτ_error; head=1, tail=1), ','))
        # save all available (if not duplicate)
        if !((i, j) in pidx)
            for k = 1:length(yf)
                if !isnan(yf[k])
                    push!(pidx, (i, j))
                    push!(y, yf[k])
                    push!(σ, σf[k])
                end
            end
        end
    end
    function differencematrix(idx, events, freq)
        D = zeros(length(idx), length(events))
        for i = 1:length(idx)
            D[i,idx[i][1]] = -1
            D[i,idx[i][2]] = 1
        end
        return repeat(D, freq)
    end
    # T-wave difference matrix
    Dt = differencematrix(tidx, events, tfreq)
    # Scholte wave difference matrix
    Ds = differencematrix(sidx, events, sfreq)
    # P-wave difference matrix
    Dp = differencematrix(pidx, events, 1)
    # origin time error matrix
    Ω = e^2*[Dt; Ds; Dp]*[Dt; Ds; Dp]'
    return y, σ, Ω
end

"""
    load hydrographicdata(data, θ)
Load the hydrographic data in `data`, represent it as `Point` measurements, and assign the uncorrelated error from `θ` corresponding to the measurement depth.
"""
function loadhydrographicdata(data, θ)
    n = size(data, 1)
    i = depthidx.(data.depth) .≤ ndepths
    y = data.temperature[i]
    σ = [θ[nparams+5ndepths+depthidx(data[j,:depth])] for j = (1:n)[i]]
    p = Point.(data.latitude[i], data.longitude[i], data.depth[i], data.time[i])
    return y, σ, p
end

function isocean(lat, lon, depth)
    h = Array{Float64}(TiffImages.load("../topo60.tiff"))
    hlon = -180 .+ (0.5:1199.5)/60
    hlat = -15 .- (0.5:599.5)/60
    ilon = argmin(abs.(lon.-hlon))
    ilat = argmin(abs.(lat.-hlat))
    return 0 ≤ depth ≤ -h[ilat,ilon]
end

# load covariance function parameters
θ = load("param_2025-08-21T22:42:32.667.jld2", "values")[end]

# load the T-wave kernels
tkernels = isfile("kernels.jld2") ? load("kernels.jld2", "tkernels") : loadkernels("../seismicdata/Twave_kernels/", 20, 3, 5.0, 50.0)
skernels = isfile("kernels.jld2") ? load("kernels.jld2", "skernels") : loadkernels("../seismicdata/Scholte_kernels/", 342, 2, 5.0, 50.0)
save("kernels.jld2", "tkernels", tkernels, "skernels", skernels)

# restrict to this time period (for now)
starttime = DateTime(1992, 3, 1)
endtime = DateTime(2025, 3, 1)

# restrict to lowest n frequencies
tfreq = 1
sfreq = 2

# load the seismic data
# tpairs = CSV.read("../seismicdata/dtTwave_with_KernelID.csv", DataFrame)
tpairs = CSV.read("../seismicdata/dtResults_Twave.csv", DataFrame)
tpairs = tpairs[starttime .< tpairs.EvtA_time .< endtime .|| starttime .< tpairs.EvtB_time .< endtime,:]
spairs = CSV.read("../seismicdata/dtResults_Scholte.csv", DataFrame)
spairs = spairs[starttime .< spairs.EvtA_time .< endtime .|| starttime .< spairs.EvtB_time .< endtime,:]
ppairs = CSV.read("../seismicdata/dtResults_Pwave.csv", DataFrame)
ppairs = ppairs[starttime .< ppairs.EvtA_time .< endtime .|| starttime .< ppairs.EvtB_time .< endtime,:]

# load the hydrographic data
hdata = CSV.read("swpacific_0500_2025-03-31.csv", DataFrame)
hdata = hdata[starttime .< hdata.time .< endtime,:]

# subsample data
Random.seed!(42)
# tpairs = tpairs[shuffle(1:size(tpairs, 1))[1:0],:]
# spairs = spairs[shuffle(1:size(spairs, 1))[1:0],:]
# idx = [ppairs.PairID[i] ∈ [tpairs.PairID; spairs.PairID] for i = 1:size(ppairs, 1)]
# ppairs = ppairs[idx,:]
tpairs = tpairs[1:0,:]
spairs = spairs[1:0,:]
ppairs = ppairs[1:0,:]
hdata = hdata[shuffle(1:size(hdata, 1))[1:16000],:]

# parse the T-wave data
yt, σt, st = loadtwavedata(tpairs, tkernels, tfreq)

# parse the Scholte wave data
ys, σs, ss = loadscholtedata(spairs, skernels, sfreq)

# parse the P-wave data
yp, σp, Ω = loadpwavedata(tpairs, spairs, ppairs, tfreq, sfreq, 3.5)

# parse the hydrographic data
yh, σh, ph = loadhydrographicdata(hdata, θ)

# observation vector (T-wave and P-wave observations)
y = [yt; ys; yp; yh]

# size of observation vectors
nt = length(yt)
ns = length(ys)
np = length(yp)
nh = length(yh)

# representation of all data (no ocean data in P waves)
r = [st; ss; fill(nothing, np); ph]

# signal covariance matrix (T waves, Scholte waves, hydrographic data)
# C = cov(covall, θ, r)

# observation covariance (signal in T wave part, plus errors and correlations)
# Ryy = C + diagm([σt; σs; σp; σh].^2) + cat(Ω, zeros(nh, nh); dims=(1, 2))
Ryy = load("cov_1992_2025_all_16k_new.jld2", "Ryy")[end-nh+1:end,end-nh+1:end]
# Ryy = load("cov_1992_2025_all_16k.jld2", "Ryy")
# Ryy = load("cov_1992_2025_all_10_16k.jld2", "Ryy", Ryy)[nt+ns+np+1:end,nt+ns+np+1:end]
# Ryy = load("cov_1990_2025_sch.jld2", "Ryy")
# Ryy = load("cov_1990_2025_bth_100_8000.jld2", "Ryy")
# Ryy = load("cov_1990_2025_bth_200_16000.jld2", "Ryy")[1:ns+np,1:ns+np]
# Ryy = load("cov_1992_2025_bth_16000.jld2", "Ryy")[1:ns+np,1:ns+np]
# Ryy = load("cov_1992_2025_bth_2f_16k.jld2", "Ryy") #[1:ns+np,1:ns+np]
# save("cov_1992_2025_all_16k_new.jld2", "Ryy", Ryy)

### TIME SERIES AT A POINT ###

# # estimation points
# t = DateTime(2020, 1, 1) : Month(1) : DateTime(2025, 1, 1)
# p = Point.(-25.0, -170.0, 1750.0, t)
# 
# # prior covariance at estimation points
# Rxx = cov(θ, p)
# 
# # covariance between estimation points and measurements
# Rxy = cov(θ, p, r)
# 
# # solve
# x = Rxy*(Ryy\y)
# P = Rxx - Rxy*(Ryy\Rxy')
# 
# # subtract mean
# D = diagm(ones(length(p)))
# D .-= 1/length(p)
# e = sqrt.(diag(D*P*D'))
# f = sqrt.(diag(D*Rxx*D'))
# 
# # plot
# figure()
# plot(t, zeros(length(t)); color="black")
# plot(t, -2f; color="black")
# plot(t, 2f; color="black")
# plot(t, D*x)
# fill_between(t, D*x - 2e, D*x + 2e; alpha=0.25)

### MAP ###

# estimation points
dl = 0.25
lat = -25.0:dl:-15
lon = -180.0:dl:-160
depth = 4750.0
p = Point.(reshape(lat.*ones(length(lon))', length(lat)*length(lon)), reshape(lon'.*ones(length(lat)), length(lat)*length(lon)), depth, reftime + Year(1))

# estimate trend
x, Rxx, P = estimate(covtrend, θ, p, r, y, Ryy)

# errors
e = sqrt.(diag(P))
f = maximum(sqrt.(diag(Rxx)))

# find ocean points
mask = isocean.(reshape(lat.*ones(length(lon))', length(lat)*length(lon)), reshape(lon'.*ones(length(lat)), length(lat)*length(lon)), depth)

# mask points below seafloor
x[.!mask] .= NaN
e[.!mask] .= NaN

extent = [lon[1]-dl/2, lon[end]+dl/2, lat[1]-dl/2, lat[end]+dl/2]

fig, ax = subplots(2, 1; sharex=true, sharey=true, figsize=(7.2, 6.4))
im1 = ax[1].imshow(reshape(1e3x, (length(lat), length(lon))); origin="lower", extent, cmap="RdBu_r", vmin=-2e3f, vmax=2e3f)
im2 = ax[2].imshow(reshape(1e3e, (length(lat), length(lon))); origin="lower", extent, cmap="Grays", vmin=0, vmax=2e3f)
colorbar(im1; ax=ax[1], shrink=0.5)
colorbar(im2; ax=ax[2], shrink=0.5)
ax[1].set_title(@sprintf("Temperature trend at %d m depth (mK\$\\,\$yr\$^{-1}\$)", depth))
ax[2].set_title("Uncertainty (K)")
fig.tight_layout()

### TREND SECTION ###

# # estimation points
# nhpos = length(skernels[33,1].hpos)
# lon = [skernels[33,1].hpos[i].lon for i = 1:nhpos]
# lat = [skernels[33,1].hpos[i].lat for i = 1:nhpos]
# depth = 250.0:500.0:5750.0
# ndepth = length(depth)
# L = hdist(skernels[33,1].hpos[1], skernels[33,1].hpos[end])
# dx = L/(nhpos-1)
# 
# lons = reshape(lon.*ones(ndepth)', nhpos*ndepth)
# lats = reshape(lat.*ones(ndepth)', nhpos*ndepth)
# depths = reshape(depth'.*ones(nhpos), nhpos*ndepth)
# 
# # points on the section
# p = Point.(lons, lats, depths, reftime + Year(1))
# 
# # estimate trend
# x, Rxx, P = estimate(covtrend, θ, p, r, y, Ryy)
# 
# # errors
# e = sqrt.(diag(P))
# f = 2.5e-3 #sqrt.(Rxx[1])
# 
# # find ocean points
# mask = isocean.(lats, lons, depths)
# 
# # average along the section
# D = kron(I(ndepth), ones(nhpos)').*mask'
# D ./= sum(D; dims=2)
# x̄ = D*x
# P̄ = D*P*D'
# 
# # mask points below seafloor
# x[.!mask] .= NaN
# e[.!mask] .= NaN
# 
# # figure setup
# extent = [L+dx/2, -dx/2, 1e-3depth[end]+0.25, 1e-3depth[1]-0.25]
# fig, ax = subplots(2, 1; sharex=true, sharey=true, figsize=(6.4, 4.8))
# 
# # plot trend and its uncertainty
# im1 = ax[1].imshow(reshape(1e3x, (nhpos, ndepth))'; extent, cmap="RdBu_r", vmin=-1e3f, vmax=1e3f, aspect="auto")
# im2 = ax[2].imshow(reshape(1e3e, (nhpos, ndepth))'; extent, cmap="Grays", vmin=0, vmax=1e3f, aspect="auto")
# colorbar(im1; ax=ax[1], shrink=0.5)
# colorbar(im2; ax=ax[2], shrink=0.5)
# ax[1].set_xlim(extent[2], extent[1])
# # ax[1].set_title(@sprintf("trend at %d°S (mK\$\\,\$yr\$^{-1}\$)", -lat))
# ax[1].set_title("trend along Scholte wave path 33 (mK\$\\,\$yr\$^{-1}\$)")
# ax[2].set_title("uncertainty (mK\$\\,\$yr\$^{-1}\$)")
# 
# fig.tight_layout()
# # fig.savefig("sec_1992_2025_20S_all_16k.pdf")
# 
# # plot trends profile
# figure()
# plot(1e3x̄, 1e-3depth)
# fill_betweenx(1e-3depth, 1e3*(x̄ - 2sqrt.(diag(P̄))), 1e3*(x̄ + 2sqrt.(diag(P̄))); linewidth=0, alpha=0.25)
# ylim(6, 0)
# xlabel("trend (mK\$\\,\$yr\$^{-1}\$)")
# ylabel("depth (km)")

### MEAN MAP ###

# # estimation points
# dl = 0.5
# lat = -25.0:dl:-15
# lon = -180.0:dl:-160
# depth = 4250.0
# nlat = length(lat)
# nlon = length(lon)
# 
# # points n the map
# p = Point.(reshape(lat.*ones(nlon)', nlat*nlon), reshape(lon'.*ones(nlat), nlat*nlon), depth, reftime + Year(1))
# 
# # figure setup
# extent = [lon[1]-dl/2, lon[end]+dl/2, lat[1]-dl/2, lat[end]+dl/2]
# fig, ax = subplots(2, 2; sharex=true, sharey=true, figsize=(12.8, 6.4))
# 
# # estimate mean
# x, Rxx, P = estimate(covmean, θ, p, r, y, Ryy)
# 
# # errors
# e = sqrt.(diag(P))
# f = sqrt.(Rxx[1])
# 
# # plot mean and its uncertainty
# im1 = ax[1,1].imshow(reshape(x, (nlat, nlon)); origin="lower", extent, cmap="RdBu_r", vmin=-2f, vmax=2f)
# im2 = ax[1,2].imshow(reshape(e, (nlat, nlon)); origin="lower", extent, cmap="Grays", vmin=0, vmax=f)
# colorbar(im1; ax=ax[1,1], shrink=0.5)
# colorbar(im2; ax=ax[1,2], shrink=0.5)
# ax[1,1].set_title(@sprintf("mean at %d m depth (°C)", depth))
# ax[1,2].set_title("uncertainty (K)")
# 
# # estimate trend
# x, Rxx, P = estimate(covtrend, θ, p, r, y, Ryy)
# 
# # errors
# e = sqrt.(diag(P))
# f = sqrt.(Rxx[1])
# 
# # plot trend and its uncertainty
# im1 = ax[2,1].imshow(reshape(1e3x, (nlat, nlon)); origin="lower", extent, cmap="RdBu_r", vmin=-2.5e3f, vmax=2.5e3f)
# im2 = ax[2,2].imshow(reshape(1e3e, (nlat, nlon)); origin="lower", extent, cmap="Grays", vmin=0, vmax=1e3f)
# colorbar(im1; ax=ax[2,1], shrink=0.5)
# colorbar(im2; ax=ax[2,2], shrink=0.5)
# ax[2,1].set_title(@sprintf("trend at %d m depth (mK\$\\,\$yr\$^{-1}\$)", depth))
# ax[2,2].set_title("uncertainty (mK\$\\,\$yr\$^{-1}\$)")
# 
# fig.tight_layout()

### TIME SERIES OF SECTION ###

# # estimation points
# t = starttime : meanmonth : endtime + meanmonth/2
# # s = Section.(tkernels[16,1], t)
# s = Section.(skernels[33,1], t)
# 
# # estimate
# x, Rxx, P = estimate(covanom, θ, s, r, y, Ryy)
# 
# # uncertainties
# e = sqrt.(diag(P))
# f = sqrt.(diag(Rxx))
# 
# # plot
# figure()
# plot(t, zeros(length(t)); color="black")
# plot(t, -2f; color="black")
# plot(t, 2f; color="black")
# plot(t, x)
# fill_between(t, x - 2e, x + 2e; alpha=0.25)
# ylabel("travel time anomaly (s)")
