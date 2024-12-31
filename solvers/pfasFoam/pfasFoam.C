/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    adsorptionFoam

Description
    ..::TODO

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "fvOptions.H"
#include "simpleControl.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
#include "setRootCaseLists.H"

#include "createTime.H"
#include "createMesh.H"

    simpleControl simple(mesh);

#include "createFields.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Foam::Info << "\nCalculating adsorption\n"
               << endl;

    while (simple.loop(runTime))
    {
        Foam::Info << "Time = " << runTime.timeName() << nl;

        while (simple.correctNonOrthogonal())
        {
            // * * * Previous calculations * * * * * * * * * * * * * * * * * //

            site_blocking = (1.0 - (pfoa_ads + pfhxa_ads + pfhxs_ads + bez_ads + dcf_ads + pfba_ads + genx_ads) / retention_capacity);

            // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

            Foam::Info << "waterPhase solver" << nl;

            // Aqueous phase of PFOA
            fvScalarMatrix Eq_pfoa_aq(
                fvm::ddt(porosity, pfoa_aq)       // d(n.C)/dt
                    + fvm::div(phi, pfoa_aq)      // q. dC/dx --Adv
                    - fvm::laplacian(DT, pfoa_aq) // D. d²C/dx²  --Diff
                ==
                -k_adsorption_pfoa * porosity * site_blocking * pfoa_aq // katt*C
                    + k_desorption_pfoa * pfoa_ads                      // kdet*S
                    + fvOptions(pfoa_aq));

            fvOptions.constrain(Eq_pfoa_aq);
            Eq_pfoa_aq.solve();
            fvOptions.correct(pfoa_aq);

            // Aqueous phase of PFHXA
            fvScalarMatrix Eq_pfhxa_aq(
                fvm::ddt(porosity, pfhxa_aq)       // d(n.C)/dt
                    + fvm::div(phi, pfhxa_aq)      // q. dC/dx --Adv
                    - fvm::laplacian(DT, pfhxa_aq) // D. d²C/dx²  --Diff
                ==
                -k_adsorption_pfhxa * porosity * site_blocking * pfhxa_aq // katt*C
                    + k_desorption_pfhxa * pfhxa_ads                      // kdet*S
                    + fvOptions(pfhxa_aq));

            fvOptions.constrain(Eq_pfhxa_aq);
            Eq_pfhxa_aq.solve();
            fvOptions.correct(pfhxa_aq);

            // Aqueous phase of PFHXS
            fvScalarMatrix Eq_pfhxs_aq(
                fvm::ddt(porosity, pfhxs_aq)       // d(n.C)/dt
                    + fvm::div(phi, pfhxs_aq)      // q. dC/dx --Adv
                    - fvm::laplacian(DT, pfhxs_aq) // d²C/dx²  --Diff
                ==
                -k_adsorption_pfhxs * porosity * site_blocking * pfhxs_aq // katt*C
                    + k_desorption_pfhxs * pfhxs_ads                      // kdet*S
                    + fvOptions(pfhxs_aq));

            fvOptions.constrain(Eq_pfhxs_aq);
            Eq_pfhxs_aq.solve();
            fvOptions.correct(pfhxs_aq);

            // Aqueous phase of BEZ
            fvScalarMatrix Eq_bez_aq(
                fvm::ddt(porosity, bez_aq)       // d(n.C)/dt
                    + fvm::div(phi, bez_aq)      // q. dC/dx --Adv
                    - fvm::laplacian(DT, bez_aq) // d²C/dx²  --Diff
                ==
                -k_adsorption_bez * porosity * site_blocking * bez_aq // katt*C
                    + k_desorption_bez * bez_ads                      // kdet*S
                    + fvOptions(bez_aq));

            fvOptions.constrain(Eq_bez_aq);
            Eq_bez_aq.solve();
            fvOptions.correct(bez_aq);

            // Aqueous phase of DCF
            fvScalarMatrix Eq_dcf_aq(
                fvm::ddt(porosity, dcf_aq)       // d(n.C)/dt
                    + fvm::div(phi, dcf_aq)      // q. dC/dx --Adv
                    - fvm::laplacian(DT, dcf_aq) // d²C/dx²  --Diff
                ==
                -k_adsorption_dcf * porosity * site_blocking * dcf_aq // katt*C
                    + k_desorption_dcf * dcf_ads                      // kdet*S
                    + fvOptions(dcf_aq));

            fvOptions.constrain(Eq_dcf_aq);
            Eq_dcf_aq.solve();
            fvOptions.correct(dcf_aq);

            // Aqueous phase of PFBA
            fvScalarMatrix Eq_pfba_aq(
                fvm::ddt(porosity, pfba_aq)       // d(n.C)/dt
                    + fvm::div(phi, pfba_aq)      // q. dC/dx --Adv
                    - fvm::laplacian(DT, pfba_aq) // d²C/dx²  --Diff
                ==
                -k_adsorption_pfba * porosity * site_blocking * pfba_aq // katt*C
                    + k_desorption_pfba * pfba_ads                      // kdet*S
                    + fvOptions(pfba_aq));

            fvOptions.constrain(Eq_pfba_aq);
            Eq_pfba_aq.solve();
            fvOptions.correct(pfba_aq);

            // Aqueous phase of GENX
            fvScalarMatrix Eq_genx_aq(
                fvm::ddt(porosity, genx_aq)       // d(n.C)/dt
                    + fvm::div(phi, genx_aq)      // q. dC/dx --Adv
                    - fvm::laplacian(DT, genx_aq) // d²C/dx²  --Diff
                ==
                -k_adsorption_genx * porosity * site_blocking * genx_aq // katt*C
                    + k_desorption_genx * genx_ads                      // kdet*S
                    + fvOptions(genx_aq));

            fvOptions.constrain(Eq_genx_aq);
            Eq_genx_aq.solve();
            fvOptions.correct(genx_aq);

            // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

            Foam::Info << "solidPhase solver" << nl;

            // Adsorbed phase of PFOA
            fvScalarMatrix Eq_pfoa_ads(
                fvm::ddt(pfoa_ads) // dS/dt
                ==
                k_adsorption_pfoa * porosity * site_blocking * pfoa_aq - k_desorption_pfoa * pfoa_ads + fvOptions(pfoa_ads));

            fvOptions.constrain(Eq_pfoa_ads);
            Eq_pfoa_ads.solve();
            fvOptions.correct(pfoa_ads);

            // Adsorbed phase of PFHXA
            fvScalarMatrix Eq_pfhxa_ads(
                fvm::ddt(pfhxa_ads) // dS/dt
                ==
                k_adsorption_pfhxa * porosity * site_blocking * pfhxa_aq - k_desorption_pfhxa * pfhxa_ads + fvOptions(pfhxa_ads));

            fvOptions.constrain(Eq_pfhxa_ads);
            Eq_pfhxa_ads.solve();
            fvOptions.correct(pfhxa_ads);

            // Adsorbed phase of PFHXS
            fvScalarMatrix Eq_pfhxs_ads(
                fvm::ddt(pfhxs_ads) // dS/dt
                ==
                k_adsorption_pfhxs * porosity * site_blocking * pfhxs_aq - k_desorption_pfhxs * pfhxs_ads + fvOptions(pfhxs_ads));

            fvOptions.constrain(Eq_pfhxs_ads);
            Eq_pfhxs_ads.solve();
            fvOptions.correct(pfhxs_ads);

            // Adsorbed phase of BEZ
            fvScalarMatrix Eq_bez_ads(
                fvm::ddt(bez_ads) // dS/dt
                ==
                k_adsorption_bez * porosity * site_blocking * bez_aq - k_desorption_bez * bez_ads + fvOptions(bez_ads));

            fvOptions.constrain(Eq_bez_ads);
            Eq_bez_ads.solve();
            fvOptions.correct(bez_ads);

            // Adsorbed phase of DCF
            fvScalarMatrix Eq_dcf_ads(
                fvm::ddt(dcf_ads) // dS/dt
                ==
                k_adsorption_dcf * porosity * site_blocking * dcf_aq - k_desorption_dcf * dcf_ads + fvOptions(dcf_ads));

            fvOptions.constrain(Eq_dcf_ads);
            Eq_dcf_ads.solve();
            fvOptions.correct(dcf_ads);

            // Adsorbed phase of PFBA
            fvScalarMatrix Eq_pfba_ads(
                fvm::ddt(pfba_ads) // dS/dt
                ==
                k_adsorption_pfba * porosity * site_blocking * pfba_aq - k_desorption_pfba * pfba_ads + fvOptions(pfba_ads));

            fvOptions.constrain(Eq_pfba_ads);
            Eq_pfba_ads.solve();
            fvOptions.correct(pfba_ads);

            // Adsorbed phase of GENX
            fvScalarMatrix Eq_genx_ads(
                fvm::ddt(genx_ads) // dS/dt
                ==
                k_adsorption_genx * porosity * site_blocking * genx_aq - k_desorption_genx * genx_ads + fvOptions(genx_ads));

            fvOptions.constrain(Eq_genx_ads);
            Eq_genx_ads.solve();
            fvOptions.correct(genx_ads);

            Foam::Info << endl;
        }

        runTime.write();

#include "CourantNo.H"

        Foam::Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
                   << "  ClockTime = " << runTime.elapsedClockTime() << " s"
                   << nl << nl;
    }

    Foam::Info << "End\n"
               << endl;

    return 0;
}

// ************************************************************************* //
