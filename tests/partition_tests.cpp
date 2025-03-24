#include "pch.h"
#include "../partition.h"
#include "../partition.cpp"  // TODO: Fix evil include to avoid linker error

using namespace npq;


// Helper function for checking that the vecIdToBlockId map is consistent with the blockIdToVecIds map
bool CheckVecIdToBlockId(const Partition& partition)
{
    for (id_t blockId = 0; blockId < partition.blockIdToVecIds.size(); ++blockId)
    {
        for (const id_t& vecId : partition.blockIdToVecIds[blockId])
        {
            if (partition.vecIdToBlockId[vecId] != blockId)
            {
				return false;
            }
        }
    }

	return true;
}


namespace PartitionTests
{

TEST(CheckVecIdToBlockIdTest, consistent)
{
    Partition p{ { 0, 1, 0 }, { { 0, 2 }, { 1 } } };

    EXPECT_TRUE(CheckVecIdToBlockId(p));
}

TEST(CheckVecIdToBlockIdTest, inconsistent)
{
    Partition p{ { 1, 0, 1 }, { { 0, 2 }, { 1 } } };

    EXPECT_FALSE(CheckVecIdToBlockId(p));
}


TEST(PartitionStructTest, Constructor_size)
{
    Partition p{ 10, 5 };

    EXPECT_EQ(p.vecIdToBlockId.size(), 10);
    EXPECT_EQ(p.blockIdToVecIds.size(), 5);
}

TEST(PartitionStructTest, Constructor_rvalue)
{
    Partition p{ { 0, 1, 0 }, { { 0, 2 }, { 1 } } };

    std::vector<id_t> vecIdToBlockId{ 0, 1, 0 };
    std::vector<std::vector<id_t>> blockIdToVecIds{ { 0, 2 }, { 1 } };
    EXPECT_EQ(p.vecIdToBlockId, vecIdToBlockId);
    EXPECT_EQ(p.blockIdToVecIds, blockIdToVecIds);
}

TEST(PartitionStructTest, equality_op_pos)
{
    Partition p{ { 0, 1, 0 }, { { 0, 2 }, { 1 } } };
    Partition q{ { 1, 0, 1 }, { { 1 }, { 2, 0 } } };

    EXPECT_EQ(p, q);
}

TEST(PartitionStructTest, equality_op_neg)
{
    Partition p{ { 0, 1, 0 }, { { 0, 2 }, { 1 } } };
    Partition q{ { 0, 0, 1 }, { { 0, 1 }, { 2 } } };

    EXPECT_NE(p, q);
}


TEST(PartitionFunctionsTest, JointPartition_match)
{
    Partition p{ { 0, 1, 0 }, { { 0, 2 }, { 1 } } };
    Partition q{ { 1, 0, 1 }, { { 1 }, { 2, 0 } } };
    Partition joint = jointPartition(p, q);

    // Check that the blockIdToVecIds maps are equivalent using the overridden equality operator
	// Note that the vecIdToBlockId map does not affect the equality check, so we omit it in the following construction
    Partition expected{ {}, { { 0, 2 }, { 1 } } };
    EXPECT_EQ(joint, expected);
	// Check that the vecIdToBlockId map is consistent with the blockIdToVecIds map
	EXPECT_TRUE(CheckVecIdToBlockId(joint));
}

TEST(PartitionFunctionsTest, JointPartition_contain) {
	Partition p{ { 0, 0, 0 }, { { 0, 1, 2 } } };
	Partition q{ { 1, 0, 1 }, { { 1 }, { 0, 2 } } };
    Partition joint = jointPartition(p, q);

	Partition expected{ {}, { { 1 }, { 0, 2 } } };
	EXPECT_EQ(joint, expected);
    EXPECT_TRUE(CheckVecIdToBlockId(joint));
}

TEST(PartitionFunctionsTest, JointPartition_straddle) {
	Partition p{ { 0, 1, 0 }, { { 0, 2 }, { 1 } } };
	Partition q{ { 0, 1, 1 }, { { 0 }, { 2, 1 } } };
    Partition joint = jointPartition(p, q);

	Partition expected{ {}, { { 0 }, { 1 }, { 2 } } };
    EXPECT_EQ(joint, expected);
    EXPECT_TRUE(CheckVecIdToBlockId(joint));
}

TEST(PartitionFunctionsTest, JointPartition_empty)
{
    Partition p{ {}, {} };
    Partition q{ {}, {} };
    Partition joint = jointPartition(p, q);

    Partition expected{ {}, {} };
    EXPECT_EQ(joint, expected);
    EXPECT_TRUE(CheckVecIdToBlockId(joint));
}

TEST(PartitionFunctionsTest, Entropy_empty) {
    Partition p{ {}, {} };
    double entropy = entropy(p);
    double expEntropy = entropy(p, true);

    double expectedEntropy = 0.0;
    double expectedExpEntropy = 1.0;
    EXPECT_NEAR(entropy, expectedEntropy, 1e-5);
    EXPECT_NEAR(expEntropy, expectedExpEntropy, 1e-5);
}

TEST(PartitionFunctionsTest, Entropy_single) {
    Partition p{ { 0, 0, 0 }, { { 0, 1, 2 } } };
    double entropy = entropy(p);
    double expEntropy = entropy(p, true);

    double expectedEntropy = 0.0;
    double expectedExpEntropy = 1.0;
    EXPECT_NEAR(entropy, expectedEntropy, 1e-5);
    EXPECT_NEAR(expEntropy, expectedExpEntropy, 1e-5);
}
TEST(PartitionFunctionsTest, Entropy_even) {
    Partition p{ { 0, 1, 2 }, { { 0 }, { 1 }, { 2 } } };
    double entropy = entropy(p);
    double expEntropy = entropy(p, true);

    double expectedEntropy = log2(3.0);
    double expectedExpEntropy = 3.0;
    EXPECT_NEAR(entropy, expectedEntropy, 1e-5);
    EXPECT_NEAR(expEntropy, expectedExpEntropy, 1e-5);
}

TEST(PartitionFunctionsTest, Entropy_uneven) {
    Partition p{ { 0, 1, 0 }, { { 0, 2 }, { 1 } } };
    double entropy = entropy(p);
    double expEntropy = entropy(p, true);

    double expectedEntropy = -2.0 / 3.0 * log2(2.0 / 3.0) - 1.0 / 3.0 * log2(1.0 / 3.0);
    double expectedExpEntropy = exp2(expectedEntropy);
    EXPECT_NEAR(entropy, expectedEntropy, 1e-5);
    EXPECT_NEAR(expEntropy, expectedExpEntropy, 1e-5);
}

TEST(PartitionFunctionsTest, Entropy_even_2) {
    Partition p{ { 0, 1, 0, 1 }, { { 0, 2 }, { 1, 3 } } };
    double entropy = entropy(p);
    double expEntropy = entropy(p, true);

    double expectedEntropy = 1.0;
    double expectedExpEntropy = 2.0;
    EXPECT_NEAR(entropy, expectedEntropy, 1e-5);
    EXPECT_NEAR(expEntropy, expectedExpEntropy, 1e-5);
}


// TODO: Add tests for the optimized composite functions

//TEST(PartitionTest, JointEntropy_) {
//}

} // namespace PartitionTests