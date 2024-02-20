Info<< "Reading scalarField `B_aq`\n" << endl;

volScalarField B_aq
(
    IOobject
    (
        "B_aq",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


Info<< "Reading scalarField `Bads`\n" << endl;

volScalarField B_ads
(
    IOobject
    (
        "B_ads",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading scalarField `C_aq`\n" << endl;

volScalarField C_aq
(
    IOobject
    (
        "C_aq",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


Info<< "Reading scalarField `Cads`\n" << endl;

volScalarField C_ads
(
    IOobject
    (
        "C_ads",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading vectorField `darcyFlux`\n" << endl;

volVectorField darcyFlux
(
    IOobject
    (
        "darcyFlux",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    ),
    mesh
);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

surfaceScalarField phi
(
    IOobject
    (
        "phi",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    fvc::flux(darcyFlux)
);

volScalarField site_blocking
(
    IOobject
    (
        "site_blocking",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("site_blocking_", dimless , 0.0)
);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

Info<< "Reading transportProperties\n" << endl;

IOdictionary transportProperties
(
    IOobject
    (
        "transportProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::NO_WRITE
    )
);

Info<< "Reading diffusivity DT\n" << endl;

dimensionedScalar DT
(
    transportProperties.lookup("DT")
);

Info<< "Reading porosity\n" << endl;

dimensionedScalar porosity
(
    transportProperties.lookup("porosity")
);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

Info<< "Reading adsorption dictionary\n" << endl;

dictionary& k_adsorption_dict(
    transportProperties.subDict("k_adsorption")
);

Info << k_adsorption_dict << endl;

dimensionedScalar k_adsorption_B
(
    k_adsorption_dict.lookup("B")
);

dimensionedScalar k_adsorption_C
(
    k_adsorption_dict.lookup("C")
);

Info<< "Reading adsorption dictionary\n" << endl;

dictionary& k_desorption_dict(
    transportProperties.subDict("k_desorption")
);

dimensionedScalar k_desorption_B
(
    k_desorption_dict.lookup("B")
);

dimensionedScalar k_desorption_C
(
    k_desorption_dict.lookup("C")
);

Info<< "Reading retention_capacity\n" << endl;

dimensionedScalar retention_capacity
(
    transportProperties.lookup("retention_capacity")
);

#include "createFvOptions.H"
