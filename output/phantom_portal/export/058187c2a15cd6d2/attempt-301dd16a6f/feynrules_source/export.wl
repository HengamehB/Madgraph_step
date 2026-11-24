mfrSourceDirectory = DirectoryName[ExpandFileName[$InputFileName]];
$HistoryLength = 0;
$FeynRulesPath = FileNameJoin[{mfrSourceDirectory, "..", "..", "..", "..", "..", "..", "vendor", "FeynRules"}];
SetDirectory[$FeynRulesPath];
FR$Parallel = False;
Get[FileNameJoin[{$FeynRulesPath, "FeynRules.m"}]];
CheckAbort[
$RecursionLimit = 2500;
$IterationLimit = 4096;
LoadModel[FileNameJoin[{mfrSourceDirectory, "SM_portable.fr"}], FileNameJoin[{mfrSourceDirectory, "extension.fr"}]];
FeynmanGauge = False;
LPortable = LSM + LNew;
If[CheckHermiticity[LPortable] =!= {}, Print["PORTABLE_HERMITICITY_FAILED"]; Quit[12]];
(* Check-only adjustment for the stock SM chiral neutrinos; exported LPortable is unchanged. *)
If[CheckKineticTermNormalisation[LPortable + GetKineticTerms[LPortable, FlavorExpand -> True, SelectParticles -> {{ve,vebar},{vm,vmbar},{vt,vtbar}}]] =!= True, Print["PORTABLE_KINETIC_CHECK_FAILED"]; Quit[13]];
If[CheckDiagonalMassTerms[LPortable] =!= True, Print["PORTABLE_MASS_CHECK_FAILED"]; Quit[14]];
portableSpectrum = GetMassSpectrum[LPortable];
If[Head[portableSpectrum] =!= TableForm, Print["PORTABLE_SPECTRUM_CHECK_FAILED"]; Quit[16]];
portableBadMasses = Select[Rest[First[portableSpectrum]], Not[TrueQ[Abs[#[[3]] - NumericalValue[Mass[#[[1]]]]] <= 10^-6 Max[1, Abs[NumericalValue[Mass[#[[1]]]]]]]] &];
If[portableBadMasses =!= {}, Print["PORTABLE_DECLARED_MASSES_MISMATCH: ", portableBadMasses]; Quit[17]];
WriteUFO[LPortable, Output -> FileNameJoin[{mfrSourceDirectory, "..", "phantom_portal_UFO"}], AddDecays -> True, RemoveGhosts -> True];
If[!FileExistsQ[FileNameJoin[{mfrSourceDirectory, "..", "phantom_portal_UFO", "vertices.py"}]], Print["PORTABLE_EXPORT_FAILED"]; Quit[15]];
Print["PORTABLE_EXPORT_SUCCESS"];
Quit[0];
, Quit[18]];
