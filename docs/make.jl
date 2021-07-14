using Documenter, AcousticMetrics

IN_CI = get(ENV, "CI", nothing)=="true"

makedocs(sitename="AcousticMetrics.jl", modules=[AcousticMetrics], doctest=false,
         format=Documenter.HTML(prettyurls=IN_CI),
         pages=["Introduction"=>"index.md"])

# if IN_CI
#     deploydocs(repo="github.com/dingraha/AcousticMetrics.jl.git", devbranch="main")
# end
