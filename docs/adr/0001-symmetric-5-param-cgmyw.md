# Implement the symmetric 5-parameter CGMYW, not the full 8-parameter asymmetric version

The attached note derives an 8-parameter model (Cn,Yn,Wn,G,Cp,Yp,Wp,M) with independent tilt/shape per tail. The project brief instead calls for "this new 5-parameter model." We resolved the mismatch by choosing the minimal reading: classical CGMY (C,G,M,Y) plus one shared power-tilt parameter W applied to both tails, giving exactly 5 parameters (C,G,M,Y,W).

Rejected alternatives: the full 8-param asymmetric CGMYW (truer to the note, but doesn't match "5-parameter" and roughly doubles the calibration's degrees of freedom — likely under-identified against a single maturity's smile); bilateral gamma per the professor's email (simplest, no special functions, but has no Y/W and doesn't match "5-parameter" either without stretching to count a drift term).

Consequence: the characteristic exponent still needs Kummer U (same functional form as the note, with Wn=Wp=W, Yn=Yp=Y, Cn=Cp=C — only G,M differ across tails, exactly like plain CGMY). Extending to the full asymmetric version later is a straightforward generalization, not a rewrite.
