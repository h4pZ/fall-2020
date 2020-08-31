# Solution to the first problem 1.
import JLD2
import Random
import CSV
import DataFrames
import FreqTables
using Statistics


dfs = DataFrames

"""
Runs the first part of the PS1.
"""
function q1()
    # 1.a
    Random.seed!(1234);

    A = Random.rand(10, 7) .* (10 - (-5)) .+ (-5);
    B = Random.randn(10, 7) .* (-15) .+ (-2);
    C = zeros(5, 7);
    C[1:5, 1:5] = A[1:5, 1:5];
    C[1:5, end - 1:end] = B[1:5, end - 1:end];
    D = A .* (A .≤ 0);

    # Printing out the arrays.
    display(A)
    display(B)
    display(C)
    display(D)

    # 1.b
    display(length(A))

    # 1.c
    display(length(unique(D)))

    # 1.d
    E = reshape(B, :, 1);
    display(E[1:10])

    # 1.e
    F = cat(A, B, dims=3)
    display(F)

    # 1.f
    F = permutedims(F, [3, 2, 1]);
    display(F)

    # 1.g
    G = kron(B, C);
    display(G)
    #H = kron(C, F)  # Doesn't work because they are of different
                     # dimensions

    # 1.h
    @JLD2.save "matrixpractice.jld" A B C D E F G

    # 1.i
    @JLD2.save "firstmatrix.jld" A B C D

    # 1.j
    C_df = DataFrames.DataFrame(C);
    CSV.write("Cmatrix.csv", C_df)

    # 1.k
    D_df = DataFrames.DataFrame(D)
    CSV.write("Dmatrix.dat", D_df; delim="|")

    return A, B, C, D
end


"""
Runs the second part of the PS1.
"""
function q2(A, B, C, D)
    # Part 2.
    # 2.a
    AB = zeros(size(A)...);

    for i = 1:size(A)[1], j = 1:size(A)[2]
        AB[i, j] = A[i, j] * B[i, j];
    end

    AB2 = A .* B;

    display(AB)
    display(AB2)

    # 2.b
    Cprime = [];

    for c in C
        if -5 ≤ c ≤ 5
            push!(Cprime, c);
        end
    end

    Cprime2 = C[-5 .≤ C .≤ 5];

    display(Cprime)
    display(Cprime2)

    # 2.c
    N = 15_169;
    K = 6;
    T = 5;

    X = zeros(N, K, T);

    # Creating the binornd since it is not in Julia base.
    binord(;n=20, p=0.6) = sum(Random.rand(n) .≤ p);

    for t in 1:T
        X[:, 1, t] .= 1;
        X[:, 2, t] = Random.rand(N, 1) .≤ 0.75 * (6 - t) / 5;
        X[:, 3, t] = Random.randn(N, 1) .* 5(t - 1) .+ (15 + t - 1);
        X[:, 4, t] = Random.randn(N, 1) .* (1 / ℯ) .+ π * (6 - t) / 3;
        X[:, 5, t] = [binord() for _ in 1:N];
        X[:, 6, t] = [binord(p=0.5) for _ in 1:N];
    end

    # 2.d
    β = zeros(K, T);
    β[1, :] = [1 + 0.25(t - 1) for t in 1:T];
    β[2, :] = [log(t) for t in 1:T];
    β[3, :] = [-sqrt(t) for t in 1:T];
    β[4, :] = [exp(t) - exp(t + 1) for t in 1:T];
    β[5, :] = [t for t in 1:T];
    β[6, :] = [t / 3 for t in 1:T];

    # 2.e
    ε = Random.randn(N, T) .* 0.36;
    Y = [sum(X[n, :, t] .* β[:, t]) + ε[n, t] for n in 1:N, t in 1:T];

    return nothing
end


"""
Runs the third part of the PS1.
"""
function q3()
    # Part 3.
    # 3.a
    data = CSV.read("nlsw88.csv");
    @JLD2.save "nlsw88.jld" data

    # 3.b
    perc_no_marriage = mean(data["married"]) * 100.0;
    println("Percentage that hasn't been married: $perc_no_marriage %")

    # 3.c
    # Alternative way.
    nobs = dfs.nrow(data);
    races = dfs.combine(dfs.groupby(data, :race), dfs.nrow => :count);
    races = dfs.transform(races, :count => x -> x * 100 / nobs);
    races = dfs.rename(races, :count_function => "percentage (%)");
    display(races)

    # Expected way (?).
    races = (FreqTables.prop ∘ FreqTables.freqtable)(data["race"]);
    display(races)

    # 3.d
    summarystats = dfs.describe(data);
    missing_grade = summarystats[summarystats["variable"] .== :grade, :]["nmissing"][1];
    println("Missing observations in the grade columns: $missing_grade")


    # 3.e
    industry_occupation = FreqTables.freqtable(data, :industry, :occupation);
    display(industry_occupation)

    # 3.f
    tab_iow = dfs.combine(dfs.groupby(data[["industry", "occupation", "wage"]],
                                      [:industry, :occupation]),
                          :wage => mean);
    display(tab_iow)
end


"""
Runs the fourth part of the PS1.
"""
function q4()
    data = CSV.read("nlsw88.csv");
    # Part 4
    # 4.a
    @JLD2.load "firstmatrix.jld" A B C D

    # 4.b, 4.c and 4.e
    """
    Takes as input a matrix A and B and returns three outputs

    i) the elemnt-by-element product of the inputs
    ii) the product A'B
    iii) the sum of all the elements of A + B
    """
    function matrixops(A, B)
        if size(A) != size(B)
            throw(DimensionMismatch("size of A not equal to size of B"))
        end
        return A .* B, A' * B, sum(A + B)
    end

    # 4.d
    display(matrixops(A, B))

    # 4.f
    try
        display(matrixops(C, D))
    catch e
        println("size of A not equal to size of B")
    end

    # 4.g
    # No need to convert it into an array.
    display(matrixops(data["ttl_exp"], data["wage"]))
end

A, B, C, D = q1();
q2(A, B, C, D);
q3();
q4();
