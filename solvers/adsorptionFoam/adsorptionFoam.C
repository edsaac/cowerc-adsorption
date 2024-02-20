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

    Info<< "\nCalculating adsorption\n" << endl;

    while (simple.loop(runTime))
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        while (simple.correctNonOrthogonal())
        {
            // * * * Previous calculations * * * * * * * * * * * * * * * * * //

            site_blocking = (1.0 - (B_ads + C_ads)/retention_capacity);

            // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

            Info<< "waterPhase solver" << nl << endl;

            fvScalarMatrix Eq_B_aq
            (
                fvm::ddt(porosity, B_aq)
                + fvm::div(phi, B_aq)
                - fvm::laplacian(DT, B_aq)
             ==
                - k_adsorption_B * porosity * site_blocking * B_aq
                + k_desorption_B * B_ads
                + fvOptions(B_aq)
            );

            fvOptions.constrain(Eq_B_aq);
            Eq_B_aq.solve();
            fvOptions.correct(B_aq);

            fvScalarMatrix Eq_C_aq
            (
                fvm::ddt(porosity, C_aq)
                + fvm::div(phi, C_aq)
                - fvm::laplacian(DT, C_aq)
             ==
                - k_adsorption_C * porosity * site_blocking * C_aq
                + k_desorption_C * C_ads
                + fvOptions(C_aq)
            );

            fvOptions.constrain(Eq_C_aq);
            Eq_C_aq.solve();
            fvOptions.correct(C_aq);
            

            // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

            Info<< "solidPhase solver" << nl << endl;

            fvScalarMatrix Eq_B_ads
            (
                fvm::ddt(B_ads)
             ==
                  k_adsorption_B * porosity * site_blocking * B_aq
                - k_desorption_B * B_ads
                + fvOptions(B_ads)
            );

            fvOptions.constrain(Eq_B_ads);
            Eq_B_ads.solve();
            fvOptions.correct(B_ads);

            fvScalarMatrix Eq_C_ads
            (
                fvm::ddt(C_ads)
             ==
                  k_adsorption_C * porosity * site_blocking * C_aq
                - k_desorption_C * C_ads
                + fvOptions(C_ads)
            );

            fvOptions.constrain(Eq_C_ads);
            Eq_C_ads.solve();
            fvOptions.correct(C_ads);
        }

        runTime.write();

        #include "CourantNo.H"

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
