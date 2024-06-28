module AMDocs
using Documenter, AcousticMetrics

function doit()
    IN_CI = get(ENV, "CI", nothing)=="true"

    makedocs(sitename="AcousticMetrics.jl", modules=[AcousticMetrics], doctest=false,
             format=Documenter.HTML(prettyurls=IN_CI),
             pages=["Introduction"=>"index.md",
                    "API"=>"api.md",
                    "Theory"=>"theory.md",
                    "Software Quality Assurance"=>"sqa.md"])

    if IN_CI
        deploydocs(repo="github.com/OpenMDAO/AcousticMetrics.jl.git", devbranch="main")
    end
end

if !isinteractive()
    doit()
end

end # module
