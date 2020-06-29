using Documenter, OILMMs

makedocs(;
    modules=[OILMMs],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/willtebbutt/OILMMs.jl/blob/{commit}{path}#L{line}",
    sitename="OILMMs.jl",
    authors="willtebbutt <wt0881@my.bristol.ac.uk>",
    assets=String[],
)

deploydocs(;
    repo="github.com/willtebbutt/OILMMs.jl",
)
