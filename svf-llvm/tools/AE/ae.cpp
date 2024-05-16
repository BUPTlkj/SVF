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

        // u32_t a = 3;
        // u32_t b = 3;

        // /// Input_x
        // IntervalMat matrix(a,b);
        // matrix(0,0) = IntervalValue(2,2);
        // matrix(0,1) = IntervalValue(1,1);
        // matrix(0,2) = IntervalValue(0,0);
        // matrix(1,0) = IntervalValue(0,0);
        // matrix(1,1) = IntervalValue(2,2);
        // matrix(1,2) = IntervalValue(0,0);

        // IntervalMatrices mats;
        // mats.push_back(matrix);

        // /// Weight normal Matrix
        // Mat matrixr_weight(a,b);
        // matrixr_weight(0,0) = 2;
        // matrixr_weight(0,1) = 1;
        // matrixr_weight(0,2) = 0;
        // matrixr_weight(1,0) = 0;
        // matrixr_weight(1,1) = 2;
        // matrixr_weight(1,2) = 0;

        // /// Biase normal Matrix
        // Mat matrixr_biase(a,b);
        // matrixr_biase(0,0) = 2;
        // matrixr_biase(0,1) = 1;
        // matrixr_biase(0,2) = 0;
        // matrixr_biase(1,0) = 0;
        // matrixr_biase(1,1) = 2;
        // matrixr_biase(1,2) = 0;

        // IntervalZ3Solver iz;
        // std::vector<std::vector<z3::expr_vector>> res_relation = iz.FullyConZ3RelationSolver(matrixr_weight, mats, matrixr_biase);

        // // 现在遍历matrixOfExprVectors以检查其内容
        // for (size_t ihh = 0; ihh < res_relation.size(); ++ihh) {
        //     std::cout << "matrixOfExprVectors 中的第 " << ihh << " 个向量：" << std::endl;
        //     for (size_t jhh = 0; jhh < res_relation[ihh].size(); ++jhh) {
        //         std::cout << "  第 " << jhh << " 个 expr_vector 包含：" << std::endl;
        //         for (unsigned khh = 0; khh < res_relation[ihh][jhh].size(); ++khh) {
        //             std::cout << "    " << res_relation[ihh][jhh][khh] << std::endl;
        //         }
        //     }
        // }



//       std::ifstream file("11.txt");
//       std::stringstream buffer;
//       buffer << file.rdbuf();
//
//       std::string str = buffer.str();
//       std::regex re("'(.*?)': \\[\\((.*?)\\), \\[(.*?)\\]\\]");
//
//       std::smatch match;
//
//       std::cout<<std::regex_search(str, match, re)<<std::endl;

       /// ONNX address
       outs()<<Options::NNName()<<std::endl;
       const std::string address = Options::NNName();

       /// DataSet address
       outs()<<Options::DataSetPath()<<std::endl;
       const std::string datapath = Options::DataSetPath();

       /// parse onnx into svf-onnx
       SVFNN svfnn(address);
       auto nodes = svfnn.get_nodes();

       /// Init nn-graph builder
       NNGraphBuilder nngraph;

       /// Init & Add node
       for (const auto& node : nodes) {
           std::visit(nngraph, node);
       }

       /// Init & Add Edge
       nngraph.AddEdges();

       /// Load dataset: mnist or cifa-10, number of dataset
       LoadData dataset(datapath, 1);
       /// Input pixel matrix
       std::pair<LabelVector, MatrixVector_3c> x = dataset.read_dataset();
       // std::cout<<"Label: "<<x.first.front()<<std::endl;

    //    double perti = 0.001;
    //    std::vector<LabelAndBounds> per_x = dataset.perturbateImages(x, perti);

       /// Run abstract interpretation on NNgraph

        Mat matrix0(8, 8);
        Mat matrix1(8, 8);
        Mat matrix2(8, 8);
        Matrices mats;

        matrix0 <<
            0.1523, 0.6152, 0.7813, 0.2013, 0.4657, 0.3122, 0.9834, 0.6745,
            0.4565, 0.1239, 0.8765, 0.7621, 0.2109, 0.5402, 0.1087, 0.9283,
            0.6827, 0.5734, 0.1908, 0.0356, 0.8964, 0.7653, 0.2109, 0.4168,
            0.9572, 0.4853, 0.7319, 0.9157, 0.1712, 0.9935, 0.5278, 0.2920,
            0.9382, 0.1763, 0.5117, 0.6216, 0.4518, 0.1380, 0.8712, 0.3928,
            0.8491, 0.2759, 0.5298, 0.1321, 0.7914, 0.6307, 0.4059, 0.7940,
            0.6555, 0.3316, 0.1270, 0.4833, 0.6741, 0.8573, 0.9946, 0.1382,
            0.2987, 0.6326, 0.9638, 0.3447, 0.1329, 0.8796, 0.9237, 0.6239;

        mats.push_back(matrix0);

        matrix1 <<
            0.7416, 0.5454, 0.0782, 0.2673, 0.3637, 0.1589, 0.1397, 0.3935,
            0.9339, 0.4017, 0.4629, 0.1104, 0.9167, 0.7280, 0.5240, 0.4681,
            0.6623, 0.1465, 0.0519, 0.8317, 0.9636, 0.4003, 0.7753, 0.7506,
            0.8249, 0.6362, 0.1959, 0.9026, 0.7696, 0.2706, 0.3796, 0.5390,
            0.2539, 0.9583, 0.8753, 0.3819, 0.6211, 0.3104, 0.0586, 0.1519,
            0.1338, 0.4104, 0.4034, 0.6226, 0.6871, 0.2581, 0.5601, 0.9425,
            0.9761, 0.8184, 0.0824, 0.3965, 0.7115, 0.5365, 0.3036, 0.8442,
            0.7255, 0.8164, 0.9452, 0.2346, 0.9093, 0.8230, 0.6509, 0.9971;

        mats.push_back(matrix1);

        matrix2 <<
            0.4352, 0.4629, 0.9906, 0.1272, 0.3095, 0.8240, 0.5988, 0.8124,
            0.2109, 0.5469, 0.7701, 0.4873, 0.1907, 0.3450, 0.3141, 0.6903,
            0.6390, 0.8795, 0.7878, 0.8185, 0.5318, 0.9087, 0.1396, 0.1650,
            0.0187, 0.7902, 0.6102, 0.1164, 0.8632, 0.4386, 0.6202, 0.4276,
            0.9163, 0.9785, 0.5454, 0.1289, 0.2046, 0.8605, 0.4218, 0.1642,
            0.2673, 0.8841, 0.6548, 0.6135, 0.4191, 0.4862, 0.1024, 0.1962,
            0.5313, 0.5920, 0.6284, 0.8210, 0.2920, 0.9228, 0.2764, 0.1430,
            0.7401, 0.7359, 0.8191, 0.6564, 0.5879, 0.2837, 0.7856, 0.3117;
        
        mats.push_back(matrix2);
        std::cout<<"Matrix size: "<<mats.size()<<std::endl;
        std::cout<<"Matrix size: "<<mats[0].rows()<<std::endl;
        std::cout<<"Matrix size: "<<mats[0].cols()<<std::endl;

        auto matt = dataset.transpose_nhw_hnw(mats);

        std::cout<<"Matrix size: "<<matt.size()<<std::endl;
        std::cout<<"Matrix size: "<<matt[0].rows()<<std::endl;
        std::cout<<"Matrix size: "<<matt[0].cols()<<std::endl;
        
       
        nngraph.Traversal(mats);
    // nngraph.Traversal(x.second.front());

    //    /// Run abstract interpretation on NNgraph Interval
    //    std::vector<std::pair<u32_t, IntervalMatrices>> in_x = dataset.convertLabelAndBoundsToIntervalMatrices(per_x) ;
    //    for(u32_t i = 0; i < in_x.size(); i++){
    //        std::cout<<in_x[i].first<<std::endl;
    //        for(const auto&intervalMat: in_x[i].second){
    //            std::cout << "IntervalMatrix :\n";
    //            std::cout << "Rows: " << intervalMat.rows() << ", Columns: " << intervalMat.cols() << "\n";
    //            for (u32_t k = 0; k < intervalMat.rows(); ++k) {
    //                for (u32_t j = 0; j < intervalMat.cols(); ++j) {
    //                    std::cout<< "[ "<< intervalMat(k, j).lb().getRealNumeral()<<", "<< intervalMat(k, j).ub().getRealNumeral() <<" ]"<< "\t";
    //                }
    //                std::cout << std::endl;
    //            }
    //            std::cout<<"****************"<<std::endl;
    //        }
    //        nngraph.IntervalTraversal(in_x[i].second);
    //    }

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