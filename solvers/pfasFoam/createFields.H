Info << "Reading scalarField `pfoa_aq`\n"
     << endl;

volScalarField pfoa_aq(
    IOobject(
        "pfoa_aq",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE),
    mesh);

Info << "Reading scalarField `pfoa_ads`\n"
     << endl;

volScalarField pfoa_ads(
    IOobject(
        "pfoa_ads",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE),
    mesh);

Info << "Reading scalarField `pfhxa_aq`\n"
     << endl;

volScalarField pfhxa_aq(
    IOobject(
        "pfhxa_aq",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE),
    mesh);

Info << "Reading scalarField `pfhxa_ads`\n"
     << endl;

volScalarField pfhxa_ads(
    IOobject(
        "pfhxa_ads",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE),
    mesh);

Info << "Reading scalarField `pfhxs_aq`\n"
     << endl;

volScalarField pfhxs_aq(
    IOobject(
        "pfhxs_aq",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE),
    mesh);

Info << "Reading scalarField `pfhxs_ads`\n"
     << endl;

volScalarField pfhxs_ads(
    IOobject(
        "pfhxs_ads",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE),
    mesh);

Info << "Reading scalarField `bez_aq`\n"
     << endl;

volScalarField bez_aq(
    IOobject(
        "bez_aq",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE),
    mesh);

Info << "Reading scalarField `bez_ads`\n"
     << endl;

volScalarField bez_ads(
    IOobject(
        "bez_ads",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE),
    mesh);

Info << "Reading scalarField `dcf_aq`\n"
     << endl;

volScalarField dcf_aq(
    IOobject(
        "dcf_aq",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE),
    mesh);

Info << "Reading scalarField `dcf_ads`\n"
     << endl;

volScalarField dcf_ads(
    IOobject(
        "dcf_ads",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE),
    mesh);

Info << "Reading scalarField `pfba_aq`\n"
     << endl;

volScalarField pfba_aq(
    IOobject(
        "pfba_aq",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE),
    mesh);

Info << "Reading scalarField `pfba_ads`\n"
     << endl;

volScalarField pfba_ads(
    IOobject(
        "pfba_ads",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE),
    mesh);

Info << "Reading scalarField `pfos_aq`\n"
     << endl;

volScalarField pfos_aq(
    IOobject(
        "pfos_aq",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE),
    mesh);

Info << "Reading scalarField `pfos_ads`\n"
     << endl;

volScalarField pfos_ads(
    IOobject(
        "pfos_ads",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE),
    mesh);

Info << "Reading vectorField `darcyFlux`\n"
     << endl;

volVectorField darcyFlux(
    IOobject(
        "darcyFlux",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE),
    mesh);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

surfaceScalarField phi(
    IOobject(
        "phi",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE),
    fvc::flux(darcyFlux));

volScalarField site_blocking(
    IOobject(
        "site_blocking",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE),
    mesh,
    dimensionedScalar("site_blocking_", dimless, 0.0));

volScalarField ads_saturation(
    IOobject(
        "ads_saturation",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE),
    mesh,
    dimensionedScalar("ads_saturation_", dimless, 0.0));

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

Info << "Reading transportProperties\n"
     << endl;

IOdictionary transportProperties(
    IOobject(
        "transportProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::NO_WRITE));

Info << "Reading diffusivity DT\n"
     << endl;

dimensionedScalar DT(
    transportProperties.lookup("DT"));

Info << "Reading porosity\n"
     << endl;

dimensionedScalar porosity(
    transportProperties.lookup("porosity"));

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

Info << "Reading adsorption dictionary\n"
     << endl;

dictionary &k_adsorption_dict(
    transportProperties.subDict("k_adsorption"));

Info << k_adsorption_dict << endl;

dimensionedScalar k_adsorption_pfoa(
    k_adsorption_dict.lookup("pfoa"));

dimensionedScalar k_adsorption_pfhxa(
    k_adsorption_dict.lookup("pfhxa"));

dimensionedScalar k_adsorption_pfhxs(
    k_adsorption_dict.lookup("pfhxs"));

dimensionedScalar k_adsorption_bez(
    k_adsorption_dict.lookup("bez"));

dimensionedScalar k_adsorption_dcf(
    k_adsorption_dict.lookup("dcf"));

dimensionedScalar k_adsorption_pfba(
    k_adsorption_dict.lookup("pfba"));

dimensionedScalar k_adsorption_pfos(
    k_adsorption_dict.lookup("pfos"));

Info << "Reading desorption dictionary\n"
     << endl;

dictionary &k_desorption_dict(
    transportProperties.subDict("k_desorption"));

dimensionedScalar k_desorption_pfoa(
    k_desorption_dict.lookup("pfoa"));

dimensionedScalar k_desorption_pfhxa(
    k_desorption_dict.lookup("pfhxa"));

dimensionedScalar k_desorption_pfhxs(
    k_desorption_dict.lookup("pfhxs"));

dimensionedScalar k_desorption_bez(
    k_desorption_dict.lookup("bez"));

dimensionedScalar k_desorption_dcf(
    k_desorption_dict.lookup("dcf"));

dimensionedScalar k_desorption_pfba(
    k_desorption_dict.lookup("pfba"));

dimensionedScalar k_desorption_pfos(
    k_desorption_dict.lookup("pfos"));

Info << "Reading retention_capacity\n"
     << endl;

dimensionedScalar retention_capacity(
    transportProperties.lookup("retention_capacity"));

#include "createFvOptions.H"
