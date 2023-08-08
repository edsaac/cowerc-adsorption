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

            site_blocking = (1.0 - (Bads + Cads)/retention_capacity);

            // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

            Info<< "waterPhase solver" << nl << endl;

            fvScalarMatrix Eq_Baq
            (
                fvm::ddt(porosity, Baq)
                + fvm::div(phi, Baq)
                - fvm::laplacian(DT, Baq)
             ==
                - k_adsorption_B * porosity * site_blocking * Baq
                + k_desorption_B * Bads
                + fvOptions(Baq)
            );

            fvOptions.constrain(Eq_Baq);
            Eq_Baq.solve();
            fvOptions.correct(Baq);

            fvScalarMatrix Eq_Caq
            (
                fvm::ddt(porosity, Caq)
                + fvm::div(phi, Caq)
                - fvm::laplacian(DT, Caq)
             ==
                - k_adsorption_C * porosity * site_blocking * Caq
                + k_desorption_C * Cads
                + fvOptions(Caq)
            );

            fvOptions.constrain(Eq_Caq);
            Eq_Caq.solve();
            fvOptions.correct(Caq);
            

            // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

            Info<< "solidPhase solver" << nl << endl;

            fvScalarMatrix Eq_Bads
            (
                fvm::ddt(Bads)
             ==
                  k_adsorption_B * porosity * site_blocking * Baq
                - k_desorption_B * Bads
                + fvOptions(Bads)
            );

            fvOptions.constrain(Eq_Bads);
            Eq_Bads.solve();
            fvOptions.correct(Bads);

            fvScalarMatrix Eq_Cads
            (
                fvm::ddt(Cads)
             ==
                  k_adsorption_C * porosity * site_blocking * Caq
                - k_desorption_C * Cads
                + fvOptions(Cads)
            );

            fvOptions.constrain(Eq_Cads);
            Eq_Cads.solve();
            fvOptions.correct(Cads);
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
