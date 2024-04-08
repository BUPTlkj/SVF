//===- ae.cpp -- Abstract Execution -------------------------------------//
//
//                     SVF: Static Value-Flow Analysis
//
// Copyright (C) <2013-2017>  <Yulei Sui>
//

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.

// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
//===-----------------------------------------------------------------------===//

/*
 // Abstract Execution
 //
 // Author: Jiawei Wang, Xiao Cheng, Jiawei Yang, Jiawei Ren, Yulei Sui
 */



#include "SVF-LLVM/SVFIRBuilder.h"
#include "WPA/WPAPass.h"
#include "Util/CommandLine.h"
#include "Util/Options.h"
#include "AE/Svfexe/ICFGSimplification.h"
#include "WPA/Andersen.h"

#include "AE/Svfexe/BufOverflowChecker.h"
#include "AE/Core/RelExeState.h"
#include "AE/Core/RelationSolver.h"

#include "IntervalZ3Solver.h"
#include "SVF-LLVM/NNgraphBuilder.h"
#include "svf-onnx/NNLoaddata.h"

using namespace SVF;
using namespace SVFUtil;


static Option<bool> SYMABS(
    "symabs",
    "symbolic abstraction test",
    false
);

static Option<bool> INNGT(
    "intervalnn",
    "interval nngraph test",
    true
);

class SymblicAbstractionTest
{
public:
    SymblicAbstractionTest() = default;

    ~SymblicAbstractionTest() = default;

    static z3::context& getContext()
    {
        return Z3Expr::getContext();
    }

    void test_print()
    {
        outs() << "hello print\n";
    }

    IntervalESBase RSY_time(IntervalESBase& inv, const Z3Expr& phi,
                            RelationSolver& rs)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        IntervalESBase resRSY = rs.RSY(inv, phi);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_time - start_time);
        outs() << "running time of RSY      : " << duration.count()
               << " microseconds\n";
        return resRSY;
    }
    IntervalESBase Bilateral_time(IntervalESBase& inv, const Z3Expr& phi,
                                  RelationSolver& rs)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        IntervalESBase resBilateral = rs.bilateral(inv, phi);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_time - start_time);
        outs() << "running time of Bilateral: " << duration.count()
               << " microseconds\n";
        return resBilateral;
    }
    IntervalESBase BS_time(IntervalESBase& inv, const Z3Expr& phi,
                           RelationSolver& rs)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        IntervalESBase resBS = rs.BS(inv, phi);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_time - start_time);
        outs() << "running time of BS       : " << duration.count()
               << " microseconds\n";
        return resBS;
    }

    void testRelExeState1_1()
    {
        outs() << sucMsg("\t SUCCESS :") << "test1_1 start\n";
        IntervalESBase itv;
        RelExeState relation;
        // var0 := [0, 1];
        itv[0] = IntervalValue(0, 1);
        relation[0] = getContext().int_const("0");
        // var1 := var0 + 1;
        relation[1] =
            getContext().int_const("1") == getContext().int_const("0") + 1;
        itv[1] = itv[0] + IntervalValue(1);
        // Test extract sub vars
        Set<u32_t> res;
        relation.extractSubVars(relation[1], res);
        assert(res == Set<u32_t>({0, 1}) && "inconsistency occurs");
        IntervalESBase inv = itv.sliceState(res);
        RelationSolver rs;
        const Z3Expr& relExpr = relation[1];
        const Z3Expr& initExpr = rs.gamma_hat(inv);
        const Z3Expr& phi = (relExpr && initExpr).simplify();
        IntervalESBase resRSY = rs.RSY(inv, phi);
        IntervalESBase resBilateral = rs.bilateral(inv, phi);
        IntervalESBase resBS = rs.BS(inv, phi);
        // 0:[0,1] 1:[1,2]
        assert(resRSY == resBS && resBS == resBilateral && "inconsistency occurs");
        for (auto r : resRSY.getVarToVal())
        {
            outs() << r.first << " " << r.second << "\n";
        }
        IntervalESBase::VarToValMap intendedRes = {{0, IntervalValue(0, 1)}, {1, IntervalValue(1, 2)}};
        assert(IntervalESBase::eqVarToValMap(resBS.getVarToVal(), intendedRes) && "inconsistency occurs");
    }

    void testRelExeState1_2()
    {
        outs() << "test1_2 start\n";
        IntervalESBase itv;
        RelExeState relation;
        // var0 := [0, 1];
        relation[0] = getContext().int_const("0");
        itv[0] = IntervalValue(0, 1);
        // var1 := var0 + 1;
        relation[1] =
            getContext().int_const("1") == getContext().int_const("0") * 2;
        itv[1] = itv[0] * IntervalValue(2);

        // Test extract sub vars
        Set<u32_t> res;
        relation.extractSubVars(relation[1], res);
        assert(res == Set<u32_t>({0, 1}) && "inconsistency occurs");
        IntervalESBase inv = itv.sliceState(res);
        RelationSolver rs;
        const Z3Expr& relExpr = relation[1];
        const Z3Expr& initExpr = rs.gamma_hat(inv);
        const Z3Expr& phi = (relExpr && initExpr).simplify();
        IntervalESBase resRSY = rs.RSY(inv, phi);
        IntervalESBase resBilateral = rs.bilateral(inv, phi);
        IntervalESBase resBS = rs.BS(inv, phi);
        // 0:[0,1] 1:[0,2]
        assert(resRSY == resBS && resBS == resBilateral && "inconsistency occurs");
        for (auto r : resRSY.getVarToVal())
        {
            outs() << r.first << " " << r.second << "\n";
        }
        IntervalESBase::VarToValMap intendedRes = {{0, IntervalValue(0, 1)}, {1, IntervalValue(0, 2)}};
        assert(IntervalESBase::eqVarToValMap(resBS.getVarToVal(), intendedRes) && "inconsistency occurs");
    }

    void testRelExeState2_1()
    {
        outs() << "test2_1 start\n";
        IntervalESBase itv;
        RelExeState relation;
        // var0 := [0, 10];
        relation[0] = getContext().int_const("0");
        itv[0] = IntervalValue(0, 10);
        // var1 := var0;
        relation[1] =
            getContext().int_const("1") == getContext().int_const("0");
        itv[1] = itv[0];
        // var2 := var1 - var0;
        relation[2] = getContext().int_const("2") ==
                      getContext().int_const("1") - getContext().int_const("0");
        itv[2] = itv[1] - itv[0];
        // Test extract sub vars
        Set<u32_t> res;
        relation.extractSubVars(relation[2], res);
        assert(res == Set<u32_t>({0, 1, 2}) && "inconsistency occurs");
        IntervalESBase inv = itv.sliceState(res);
        RelationSolver rs;
        const Z3Expr& relExpr = relation[2] && relation[1];
        const Z3Expr& initExpr = rs.gamma_hat(inv);
        const Z3Expr& phi = (relExpr && initExpr).simplify();
        IntervalESBase resRSY = rs.RSY(inv, phi);
        IntervalESBase resBilateral = rs.bilateral(inv, phi);
        IntervalESBase resBS = rs.BS(inv, phi);
        // 0:[0,10] 1:[0,10] 2:[0,0]
        assert(resRSY == resBS && resBS == resBilateral && "inconsistency occurs");
        for (auto r : resRSY.getVarToVal())
        {
            outs() << r.first << " " << r.second << "\n";
        }
        // ground truth
        IntervalESBase::VarToValMap intendedRes = {{0, IntervalValue(0, 10)},
            {1, IntervalValue(0, 10)},
            {2, IntervalValue(0, 0)}
        };
        assert(IntervalESBase::eqVarToValMap(resBS.getVarToVal(), intendedRes) && "inconsistency occurs");
    }

    void testRelExeState2_2()
    {
        outs() << "test2_2 start\n";
        IntervalESBase itv;
        RelExeState relation;
        // var0 := [0, 100];
        relation[0] = getContext().int_const("0");
        itv[0] = IntervalValue(0, 100);
        // var1 := var0;
        relation[1] =
            getContext().int_const("1") == getContext().int_const("0");
        itv[1] = itv[0];
        // var2 := var1 - var0;
        relation[2] = getContext().int_const("2") ==
                      getContext().int_const("1") - getContext().int_const("0");
        itv[2] = itv[1] - itv[0];

        // Test extract sub vars
        Set<u32_t> res;
        relation.extractSubVars(relation[2], res);
        assert(res == Set<u32_t>({0, 1, 2}) && "inconsistency occurs");
        IntervalESBase inv = itv.sliceState(res);
        RelationSolver rs;
        const Z3Expr& relExpr = relation[2] && relation[1];
        const Z3Expr& initExpr = rs.gamma_hat(inv);
        const Z3Expr& phi = (relExpr && initExpr).simplify();
        IntervalESBase resRSY = rs.RSY(inv, phi);
        IntervalESBase resBilateral = rs.bilateral(inv, phi);
        IntervalESBase resBS = rs.BS(inv, phi);
        // 0:[0,100] 1:[0,100] 2:[0,0]
        assert(resRSY == resBS && resBS == resBilateral && "inconsistency occurs");
        for (auto r : resRSY.getVarToVal())
        {
            outs() << r.first << " " << r.second << "\n";
        }
        // ground truth
        IntervalESBase::VarToValMap intendedRes = {{0, IntervalValue(0, 100)},
            {1, IntervalValue(0, 100)},
            {2, IntervalValue(0, 0)}
        };
        assert(IntervalESBase::eqVarToValMap(resBS.getVarToVal(), intendedRes) && "inconsistency occurs");
    }

    void testRelExeState2_3()
    {
        outs() << "test2_3 start\n";
        IntervalESBase itv;
        RelExeState relation;
        // var0 := [0, 1000];
        relation[0] = getContext().int_const("0");
        itv[0] = IntervalValue(0, 1000);
        // var1 := var0;
        relation[1] =
            getContext().int_const("1") == getContext().int_const("0");
        itv[1] = itv[0];
        // var2 := var1 - var0;
        relation[2] = getContext().int_const("2") ==
                      getContext().int_const("1") - getContext().int_const("0");
        itv[2] = itv[1] - itv[0];

        // Test extract sub vars
        Set<u32_t> res;
        relation.extractSubVars(relation[2], res);
        assert(res == Set<u32_t>({0, 1, 2}) && "inconsistency occurs");
        IntervalESBase inv = itv.sliceState(res);
        RelationSolver rs;
        const Z3Expr& relExpr = relation[2] && relation[1];
        const Z3Expr& initExpr = rs.gamma_hat(inv);
        const Z3Expr& phi = (relExpr && initExpr).simplify();
        IntervalESBase resRSY = rs.RSY(inv, phi);
        IntervalESBase resBilateral = rs.bilateral(inv, phi);
        IntervalESBase resBS = rs.BS(inv, phi);
        // 0:[0,1000] 1:[0,1000] 2:[0,0]
        assert(resRSY == resBS && resBS == resBilateral && "inconsistency occurs");
        for (auto r : resRSY.getVarToVal())
        {
            outs() << r.first << " " << r.second << "\n";
        }
        // ground truth
        // ground truth
        IntervalESBase::VarToValMap intendedRes = {{0, IntervalValue(0, 1000)},
            {1, IntervalValue(0, 1000)},
            {2, IntervalValue(0, 0)}
        };
        assert(IntervalESBase::eqVarToValMap(resBS.getVarToVal(), intendedRes) && "inconsistency occurs");
    }

    void testRelExeState2_4()
    {
        outs() << "test2_4 start\n";
        IntervalESBase itv;
        RelExeState relation;
        // var0 := [0, 10000];
        relation[0] = getContext().int_const("0");
        itv[0] = IntervalValue(0, 10000);
        // var1 := var0;
        relation[1] =
            getContext().int_const("1") == getContext().int_const("0");
        itv[1] = itv[0];
        // var2 := var1 - var0;
        relation[2] = getContext().int_const("2") ==
                      getContext().int_const("1") - getContext().int_const("0");
        itv[2] = itv[1] - itv[0];

        // Test extract sub vars
        Set<u32_t> res;
        relation.extractSubVars(relation[2], res);
        assert(res == Set<u32_t>({0, 1, 2}) && "inconsistency occurs");
        IntervalESBase inv = itv.sliceState(res);
        RelationSolver rs;
        const Z3Expr& relExpr = relation[2] && relation[1];
        const Z3Expr& initExpr = rs.gamma_hat(inv);
        const Z3Expr& phi = (relExpr && initExpr).simplify();
        IntervalESBase resRSY = RSY_time(inv, phi, rs);
        IntervalESBase resBilateral = Bilateral_time(inv, phi, rs);
        IntervalESBase resBS = BS_time(inv, phi, rs);
        // 0:[0,10000] 1:[0,10000] 2:[0,0]
        assert(resRSY == resBS && resBS == resBilateral && "inconsistency occurs");
        for (auto r : resRSY.getVarToVal())
        {
            outs() << r.first << " " << r.second << "\n";
        }
        // ground truth
        // ground truth
        IntervalESBase::VarToValMap intendedRes = {{0, IntervalValue(0, 10000)},
            {1, IntervalValue(0, 10000)},
            {2, IntervalValue(0, 0)}
        };
        assert(IntervalESBase::eqVarToValMap(resBS.getVarToVal(), intendedRes) && "inconsistency occurs");
    }

    void testRelExeState2_5()
    {
        outs() << "test2_5 start\n";
        IntervalESBase itv;
        RelExeState relation;
        // var0 := [0, 100000];
        relation[0] = getContext().int_const("0");
        itv[0] = IntervalValue(0, 100000);
        // var1 := var0;
        relation[1] =
            getContext().int_const("1") == getContext().int_const("0");
        itv[1] = itv[0];
        // var2 := var1 - var0;
        relation[2] = getContext().int_const("2") ==
                      getContext().int_const("1") - getContext().int_const("0");
        itv[2] = itv[1] - itv[0];

        // Test extract sub vars
        Set<u32_t> res;
        relation.extractSubVars(relation[2], res);
        assert(res == Set<u32_t>({0, 1, 2}) && "inconsistency occurs");
        IntervalESBase inv = itv.sliceState(res);
        RelationSolver rs;
        const Z3Expr& relExpr = relation[2] && relation[1];
        const Z3Expr& initExpr = rs.gamma_hat(inv);
        const Z3Expr& phi = (relExpr && initExpr).simplify();
        IntervalESBase resRSY = RSY_time(inv, phi, rs);
        IntervalESBase resBilateral = Bilateral_time(inv, phi, rs);
        IntervalESBase resBS = BS_time(inv, phi, rs);
        // 0:[0,100000] 1:[0,100000] 2:[0,0]
        assert(resRSY == resBS && resBS == resBilateral && "inconsistency occurs");
        for (auto r : resRSY.getVarToVal())
        {
            outs() << r.first << " " << r.second << "\n";
        }
        // ground truth
        // ground truth
        IntervalESBase::VarToValMap intendedRes = {{0, IntervalValue(0, 100000)},
            {1, IntervalValue(0, 100000)},
            {2, IntervalValue(0, 0)}
        };
        assert(IntervalESBase::eqVarToValMap(resBS.getVarToVal(), intendedRes) && "inconsistency occurs");
    }

    void testRelExeState3_1()
    {
        outs() << "test3_1 start\n";
        IntervalESBase itv;
        RelExeState relation;
        // var0 := [1, 10];
        relation[0] = getContext().int_const("0");
        itv[0] = IntervalValue(1, 10);
        // var1 := var0;
        relation[1] =
            getContext().int_const("1") == getContext().int_const("0");
        itv[1] = itv[0];
        // var2 := var1 / var0;
        relation[2] = getContext().int_const("2") ==
                      getContext().int_const("1") / getContext().int_const("0");
        itv[2] = itv[1] / itv[0];
        // Test extract sub vars
        Set<u32_t> res;
        relation.extractSubVars(relation[2], res);
        assert(res == Set<u32_t>({0, 1, 2}) && "inconsistency occurs");
        IntervalESBase inv = itv.sliceState(res);
        RelationSolver rs;
        const Z3Expr& relExpr = relation[2] && relation[1];
        const Z3Expr& initExpr = rs.gamma_hat(inv);
        const Z3Expr& phi = (relExpr && initExpr).simplify();
        IntervalESBase resRSY = rs.RSY(inv, phi);
        IntervalESBase resBilateral = rs.bilateral(inv, phi);
        IntervalESBase resBS = rs.BS(inv, phi);
        // 0:[1,10] 1:[1,10] 2:[1,1]
        assert(resRSY == resBS && resBS == resBilateral && "inconsistency occurs");
        for (auto r : resRSY.getVarToVal())
        {
            outs() << r.first << " " << r.second << "\n";
        }
        // ground truth
        IntervalESBase::VarToValMap intendedRes = {{0, IntervalValue(1, 10)},
            {1, IntervalValue(1, 10)},
            {2, IntervalValue(1, 1)}
        };
        assert(IntervalESBase::eqVarToValMap(resBS.getVarToVal(), intendedRes) && "inconsistency occurs");
    }

    void testRelExeState3_2()
    {
        outs() << "test3_2 start\n";
        IntervalESBase itv;
        RelExeState relation;
        // var0 := [1, 1000];
        relation[0] = getContext().int_const("0");
        itv[0] = IntervalValue(1, 1000);
        // var1 := var0;
        relation[1] =
            getContext().int_const("1") == getContext().int_const("0");
        itv[1] = itv[0];
        // var2 := var1 / var0;
        relation[2] = getContext().int_const("2") ==
                      getContext().int_const("1") / getContext().int_const("0");
        itv[2] = itv[1] / itv[0];
        // Test extract sub vars
        Set<u32_t> res;
        relation.extractSubVars(relation[2], res);
        assert(res == Set<u32_t>({0, 1, 2}) && "inconsistency occurs");
        IntervalESBase inv = itv.sliceState(res);
        RelationSolver rs;
        const Z3Expr& relExpr = relation[2] && relation[1];
        const Z3Expr& initExpr = rs.gamma_hat(inv);
        const Z3Expr& phi = (relExpr && initExpr).simplify();
        IntervalESBase resRSY = rs.RSY(inv, phi);
        IntervalESBase resBilateral = rs.bilateral(inv, phi);
        IntervalESBase resBS = rs.BS(inv, phi);
        // 0:[1,1000] 1:[1,1000] 2:[1,1]
        assert(resRSY == resBS && resBS == resBilateral && "inconsistency occurs");
        for (auto r : resRSY.getVarToVal())
        {
            outs() << r.first << " " << r.second << "\n";
        }
        // ground truth
        IntervalESBase::VarToValMap intendedRes = {{0, IntervalValue(1, 1000)},
            {1, IntervalValue(1, 1000)},
            {2, IntervalValue(1, 1)}
        };
        assert(IntervalESBase::eqVarToValMap(resBS.getVarToVal(), intendedRes) && "inconsistency occurs");
    }

    void testRelExeState3_3()
    {
        outs() << "test3_3 start\n";
        IntervalESBase itv;
        RelExeState relation;
        // var0 := [1, 10000];
        relation[0] = getContext().int_const("0");
        itv[0] = IntervalValue(1, 10000);
        // var1 := var0;
        relation[1] =
            getContext().int_const("1") == getContext().int_const("0");
        itv[1] = itv[0];
        // var2 := var1 / var0;
        relation[2] = getContext().int_const("2") ==
                      getContext().int_const("1") / getContext().int_const("0");
        itv[2] = itv[1] / itv[0];
        // Test extract sub vars
        Set<u32_t> res;
        relation.extractSubVars(relation[2], res);
        assert(res == Set<u32_t>({0, 1, 2}) && "inconsistency occurs");
        IntervalESBase inv = itv.sliceState(res);
        RelationSolver rs;
        const Z3Expr& relExpr = relation[2] && relation[1];
        const Z3Expr& initExpr = rs.gamma_hat(inv);
        const Z3Expr& phi = (relExpr && initExpr).simplify();
        IntervalESBase resRSY = RSY_time(inv, phi, rs);
        IntervalESBase resBilateral = Bilateral_time(inv, phi, rs);
        IntervalESBase resBS = BS_time(inv, phi, rs);
        // 0:[1,10000] 1:[1,10000] 2:[1,1]
        assert(resRSY == resBS && resBS == resBilateral && "inconsistency occurs");
        for (auto r : resRSY.getVarToVal())
        {
            outs() << r.first << " " << r.second << "\n";
        }
        // ground truth
        IntervalESBase::VarToValMap intendedRes =
        Map<u32_t, IntervalValue>({{0, IntervalValue(1, 10000)},
            {1, IntervalValue(1, 10000)},
            {2, IntervalValue(1, 1)}});
    }

    void testRelExeState3_4()
    {
        outs() << "test3_4 start\n";
        IntervalESBase itv;
        RelExeState relation;
        // var0 := [1, 100000];
        relation[0] = getContext().int_const("0");
        itv[0] = IntervalValue(1, 100000);
        // var1 := var0;
        relation[1] =
            getContext().int_const("1") == getContext().int_const("0");
        itv[1] = itv[0];
        // var2 := var1 / var0;
        relation[2] = getContext().int_const("2") ==
                      getContext().int_const("1") / getContext().int_const("0");
        itv[2] = itv[1] / itv[0];
        // Test extract sub vars
        Set<u32_t> res;
        relation.extractSubVars(relation[2], res);
        assert(res == Set<u32_t>({0, 1, 2}) && "inconsistency occurs");
        IntervalESBase inv = itv.sliceState(res);
        RelationSolver rs;
        const Z3Expr& relExpr = relation[2] && relation[1];
        const Z3Expr& initExpr = rs.gamma_hat(inv);
        const Z3Expr& phi = (relExpr && initExpr).simplify();
        IntervalESBase resRSY = RSY_time(inv, phi, rs);
        IntervalESBase resBilateral = Bilateral_time(inv, phi, rs);
        IntervalESBase resBS = BS_time(inv, phi, rs);
        // 0:[1,100000] 1:[1,100000] 2:[1,1]
        assert(resRSY == resBS && resBS == resBilateral && "inconsistency occurs");
        for (auto r : resRSY.getVarToVal())
        {
            outs() << r.first << " " << r.second << "\n";
        }
        // ground truth
        IntervalESBase::VarToValMap intendedRes = {{0, IntervalValue(1, 100000)},
            {1, IntervalValue(1, 100000)},
            {2, IntervalValue(1, 1)}
        };
        assert(IntervalESBase::eqVarToValMap(resBS.getVarToVal(), intendedRes) && "inconsistency occurs");
    }

    void testRelExeState4_1()
    {
        outs() << "test4_1 start\n";
        IntervalESBase itv;
        RelExeState relation;
        // var0 := [0, 10];
        relation[0] = getContext().int_const("0");
        itv[0] = IntervalValue(0, 10);
        // var1 := var0;
        relation[1] =
            getContext().int_const("1") == getContext().int_const("0");
        itv[1] = itv[0];
        // var2 := var1 / var0;
        relation[2] = getContext().int_const("2") ==
                      getContext().int_const("1") / getContext().int_const("0");
        itv[2] = itv[1] / itv[0];
        // Test extract sub vars
        Set<u32_t> res;
        relation.extractSubVars(relation[2], res);
        assert(res == Set<u32_t>({0, 1, 2}) && "inconsistency occurs");
        IntervalESBase inv = itv.sliceState(res);
        RelationSolver rs;
        const Z3Expr& relExpr = relation[2] && relation[1];
        const Z3Expr& initExpr = rs.gamma_hat(inv);
        const Z3Expr& phi = (relExpr && initExpr).simplify();
        // IntervalExeState resRSY = rs.RSY(inv, phi);
        outs() << "rsy done\n";
        // IntervalExeState resBilateral = rs.bilateral(inv, phi);
        outs() << "bilateral done\n";
        IntervalESBase resBS = rs.BS(inv, phi);
        outs() << "bs done\n";
        // 0:[0,10] 1:[0,10] 2:[-00,+00]
        // assert(resRSY == resBS && resBS == resBilateral);
        for (auto r : resBS.getVarToVal())
        {
            outs() << r.first << " " << r.second << "\n";
        }
        // ground truth
        IntervalESBase::VarToValMap intendedRes = {{0, IntervalValue(0, 10)},
            {1, IntervalValue(0, 10)},
            {2, IntervalValue(IntervalValue::minus_infinity(), IntervalValue::plus_infinity())}
        };
        assert(IntervalESBase::eqVarToValMap(resBS.getVarToVal(), intendedRes) && "inconsistency occurs");
    }

    void testsValidation()
    {
        SymblicAbstractionTest saTest;
        saTest.testRelExeState1_1();
        saTest.testRelExeState1_2();

        saTest.testRelExeState2_1();
        saTest.testRelExeState2_2();
        saTest.testRelExeState2_3();
        //        saTest.testRelExeState2_4(); /// 10000
        //        saTest.testRelExeState2_5(); /// 100000

        saTest.testRelExeState3_1();
        saTest.testRelExeState3_2();
        //        saTest.testRelExeState3_3(); /// 10000
        //        saTest.testRelExeState3_4(); /// 100000

        outs() << "start top\n";
        saTest.testRelExeState4_1(); /// top
    }
};

int main(int argc, char** argv)
{
    int arg_num = 0;
    int extraArgc = 3;
    char **arg_value = new char *[argc + extraArgc];
    for (; arg_num < argc; ++arg_num)
    {
        arg_value[arg_num] = argv[arg_num];
    }
    // add extra options
    arg_value[arg_num++] = (char*) "-model-consts=true";
    arg_value[arg_num++] = (char*) "-model-arrays=true";
    arg_value[arg_num++] = (char*) "-pre-field-sensitive=false";
    assert(arg_num == (argc + extraArgc) && "more extra arguments? Change the value of extraArgc");

    std::vector<std::string> moduleNameVec;
    moduleNameVec = OptionBase::parseOptions(
                        arg_num, arg_value, "Static Symbolic Execution", "[options] <input-bitcode...>"
                    );
    delete[] arg_value;
    if (SYMABS())
    {
        SymblicAbstractionTest saTest;
        saTest.testsValidation();
        return 0;
    }else if(INNGT()){

        u32_t a = 3;
        u32_t b = 3;

        /// Input_x
        IntervalMat matrix(a,b);
        matrix(0,0) = IntervalValue(2,2);
        matrix(0,1) = IntervalValue(1,1);
        matrix(0,2) = IntervalValue(0,0);
        matrix(1,0) = IntervalValue(0,0);
        matrix(1,1) = IntervalValue(2,2);
        matrix(1,2) = IntervalValue(0,0);

        IntervalMatrices mats;
        mats.push_back(matrix);

        /// Weight normal Matrix
        Mat matrixr_weight(a,b);
        matrixr_weight(0,0) = 2;
        matrixr_weight(0,1) = 1;
        matrixr_weight(0,2) = 0;
        matrixr_weight(1,0) = 0;
        matrixr_weight(1,1) = 2;
        matrixr_weight(1,2) = 0;

        /// Biase normal Matrix
        Mat matrixr_biase(a,b);
        matrixr_biase(0,0) = 2;
        matrixr_biase(0,1) = 1;
        matrixr_biase(0,2) = 0;
        matrixr_biase(1,0) = 0;
        matrixr_biase(1,1) = 2;
        matrixr_biase(1,2) = 0;


        IntervalZ3Solver iz;
        iz.Multiply(matrixr_weight, mats, matrixr_biase);



//        std::ifstream file("11.txt");
//        std::stringstream buffer;
//        buffer << file.rdbuf();
//
//        std::string str = buffer.str();
//        std::regex re("'(.*?)': \\[\\((.*?)\\), \\[(.*?)\\]\\]");
//
//        std::smatch match;
//
//        std::cout<<std::regex_search(str, match, re)<<std::endl;
//
//        /// ONNX address
//        outs()<<Options::NNName();
//        const std::string address = Options::NNName();
//
//        /// DataSet address
//        outs()<<Options::DataSetPath();
//        const std::string datapath = Options::DataSetPath();
//
//        /// parse onnx into svf-onnx
//        SVFNN svfnn(address);
//        auto nodes = svfnn.get_nodes();
//
//        /// Init nn-graph builder
//        NNGraphBuilder nngraph;
//
//        /// Init & Add node
//        for (const auto& node : nodes) {
//            std::visit(nngraph, node);
//        }
//
//        /// Init & Add Edge
//        nngraph.AddEdges();
//
//        /// Load dataset: mnist or cifa-10, number of dataset
//        LoadData dataset(datapath, 1);
//        /// Input pixel matrix
//        std::pair<LabelVector, MatrixVector_3c> x = dataset.read_dataset();
//        std::cout<<"Label: "<<x.first.front()<<std::endl;
//
//        double perti = 0.001;
//        std::vector<LabelAndBounds> per_x = dataset.perturbateImages(x, perti);
//
//        /// Run abstract interpretation on NNgraph
//        nngraph.Traversal(x.second.front());
//
//        /// Run abstract interpretation on NNgraph Interval
//        std::vector<std::pair<u32_t, IntervalMatrices>> in_x = dataset.convertLabelAndBoundsToIntervalMatrices(per_x) ;
//        for(u32_t i = 0; i < in_x.size(); i++){
//            std::cout<<in_x[i].first<<std::endl;
//            for(const auto&intervalMat: in_x[i].second){
//                std::cout << "IntervalMatrix :\n";
//                std::cout << "Rows: " << intervalMat.rows() << ", Columns: " << intervalMat.cols() << "\n";
//                for (u32_t k = 0; k < intervalMat.rows(); ++k) {
//                    for (u32_t j = 0; j < intervalMat.cols(); ++j) {
//                        std::cout<< "[ "<< intervalMat(k, j).lb().getRealNumeral()<<", "<< intervalMat(k, j).ub().getRealNumeral() <<" ]"<< "\t";
//                    }
//                    std::cout << std::endl;
//                }
//                std::cout<<"****************"<<std::endl;
//            }
//            nngraph.IntervalTraversal(in_x[i].second);
//        }

        return 0;
    }

    SVFModule *svfModule = LLVMModuleSet::getLLVMModuleSet()->buildSVFModule(moduleNameVec);
    SVFIRBuilder builder(svfModule);
    SVFIR* pag = builder.build();
    AndersenWaveDiff* ander = AndersenWaveDiff::createAndersenWaveDiff(pag);
    PTACallGraph* callgraph = ander->getPTACallGraph();
    builder.updateCallGraph(callgraph);
    pag->getICFG()->updateCallGraph(callgraph);
    if (Options::ICFGMergeAdjacentNodes())
    {
        ICFGSimplification::mergeAdjacentNodes(pag->getICFG());
    }

    if (Options::BufferOverflowCheck())
    {
        BufOverflowChecker ae;
        ae.runOnModule(pag->getICFG());
    }
    else
    {
        AbstractExecution ae;

        ae.runOnModule(pag->getICFG());
    }

    LLVMModuleSet::releaseLLVMModuleSet();

    return 0;
}