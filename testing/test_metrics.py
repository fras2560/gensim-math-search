'''
Name: Dallas Fraser
Email: d6fraser@uwaterloo.ca
Date: 2017-07-27
Project: Tangent GT
Purpose: Tests various metrics
'''
from testing.test_pipeline import TestIndexer
import unittest
from ranking.metrics import jensen_shannon_divergence, hellinger_distance


class Test(TestIndexer):
    def testJensenShannonDivergenceLDA(self):
        # different metrices
        # the different results
        vec1 = "Human machine interface for lab abc computer applications"
        vec1 = vec1.lower().strip().split()
        vec1 = self.dictionary.doc2bow(vec1)
        vec1 = self.lda[vec1]
        vec2 = "A survey of user opinion of computer system response time"
        vec2 = vec2.lower().strip().split()
        vec2 = self.dictionary.doc2bow(vec2)
        vec2 = self.lda[vec2]
        vec3 = "Graph minors IV Widths of trees and well quasi ordering"
        vec3 = vec3.lower().strip().split()
        vec3 = self.dictionary.doc2bow(vec3)
        vec3 = self.lda[vec3]
        # check the similarities
        sim = jensen_shannon_divergence(vec1, vec2, self.lda)
        self.log(sim)
        self.assertAlmostEqual(sim, 0.10, delta=0.2)
        sim = jensen_shannon_divergence(vec1, vec3, self.lda)
        self.log(sim)
        self.assertAlmostEqual(sim, 0.10, delta=0.2)
        sim = jensen_shannon_divergence(vec2, vec3, self.lda)
        self.log(sim)
        self.assertAlmostEqual(sim, 0.10, delta=0.2)
        # compare against itself
        sim = jensen_shannon_divergence(vec1, vec1, self.lda)
        self.log(sim)
        self.assertAlmostEqual(sim, 0.0, delta=0.0002)
        sim = jensen_shannon_divergence(vec2, vec2, self.lda)
        self.log(sim)
        self.assertAlmostEqual(sim, 0.0, delta=0.0002)
        sim = jensen_shannon_divergence(vec3, vec3, self.lda)
        self.log(sim)
        self.assertAlmostEqual(sim, 0.0, delta=0.0002)

    def testJensenShannonDivergenceHDP(self):
        # different metrices
        # the different results
        vec1 = "Human machine interface for lab abc computer applications"
        vec1 = vec1.lower().strip().split()
        vec1 = self.dictionary.doc2bow(vec1)
        vec1 = self.hdp[vec1]
        vec2 = "A survey of user opinion of computer system response time"
        vec2 = vec2.lower().strip().split()
        vec2 = self.dictionary.doc2bow(vec2)
        vec2 = self.hdp[vec2]
        vec3 = "Graph minors IV Widths of trees and well quasi ordering"
        vec3 = vec3.lower().strip().split()
        vec3 = self.dictionary.doc2bow(vec3)
        vec3 = self.hdp[vec3]
        # check the similarities
        sim = jensen_shannon_divergence(vec1, vec2, self.hdp)
        self.log(sim)
        self.assertAlmostEqual(sim, 0.30, delta=0.3)
        sim = jensen_shannon_divergence(vec1, vec3, self.hdp)
        self.log(sim)
        self.assertAlmostEqual(sim, 0.30, delta=0.3)
        sim = jensen_shannon_divergence(vec2, vec3, self.hdp)
        self.log(sim)
        self.assertAlmostEqual(sim, 0.30, delta=0.3)
        # compare against itself
        sim = jensen_shannon_divergence(vec1, vec1, self.hdp)
        self.log(sim)
        self.assertAlmostEqual(sim, 0.0, delta=0.0002)
        sim = jensen_shannon_divergence(vec2, vec2, self.hdp)
        self.log(sim)
        self.assertAlmostEqual(sim, 0.0, delta=0.0002)
        sim = jensen_shannon_divergence(vec3, vec3, self.hdp)
        self.log(sim)
        self.assertAlmostEqual(sim, 0.0, delta=0.0002)

    def testHellingerDistanceLDA(self):
        # different metrices
        # the different results
        vec1 = "Human machine interface for lab abc computer applications"
        vec1 = vec1.lower().strip().split()
        vec1 = self.dictionary.doc2bow(vec1)
        vec1 = self.lda[vec1]
        vec2 = "A survey of user opinion of computer system response time"
        vec2 = vec2.lower().strip().split()
        vec2 = self.dictionary.doc2bow(vec2)
        vec2 = self.lda[vec2]
        vec3 = "Graph minors IV Widths of trees and well quasi ordering"
        vec3 = vec3.lower().strip().split()
        vec3 = self.dictionary.doc2bow(vec3)
        vec3 = self.lda[vec3]
        # check the similarities
        sim = hellinger_distance(vec1, vec2, self.lda)
        self.log(sim)
        self.assertAlmostEqual(sim, 0.30, delta=0.3)
        sim = hellinger_distance(vec1, vec3, self.lda)
        self.log(sim)
        self.assertAlmostEqual(sim, 0.30, delta=0.3)
        sim = hellinger_distance(vec2, vec3, self.lda)
        self.log(sim)
        self.assertAlmostEqual(sim, 0.30, delta=0.3)
        # compare against itself
        sim = hellinger_distance(vec1, vec1, self.lda)
        self.log(sim)
        self.assertAlmostEqual(sim, 0.0, delta=0.0002)
        sim = hellinger_distance(vec2, vec2, self.lda)
        self.log(sim)
        self.assertAlmostEqual(sim, 0.0, delta=0.0002)
        sim = hellinger_distance(vec3, vec3, self.lda)
        self.log(sim)
        self.assertAlmostEqual(sim, 0.0, delta=0.0002)

    def testHellingerDistanceHDP(self):
        # different metrices
        # the different results
        vec1 = "Human machine interface for lab abc computer applications"
        vec1 = vec1.lower().strip().split()
        vec1 = self.dictionary.doc2bow(vec1)
        vec1 = self.hdp[vec1]
        vec2 = "A survey of user opinion of computer system response time"
        vec2 = vec2.lower().strip().split()
        vec2 = self.dictionary.doc2bow(vec2)
        vec2 = self.hdp[vec2]
        vec3 = "Graph minors IV Widths of trees and well quasi ordering"
        vec3 = vec3.lower().strip().split()
        vec3 = self.dictionary.doc2bow(vec3)
        vec3 = self.hdp[vec3]
        # check the similarities
        sim = hellinger_distance(vec1, vec2, self.hdp)
        self.log(sim)
        self.assertAlmostEqual(sim, 0.35, delta=0.3)
        sim = hellinger_distance(vec1, vec3, self.hdp)
        self.log(sim)
        self.assertAlmostEqual(sim, 0.30, delta=0.3)
        sim = hellinger_distance(vec2, vec3, self.hdp)
        self.log(sim)
        self.assertAlmostEqual(sim, 0.30, delta=0.3)
        # compare against itself
        sim = hellinger_distance(vec1, vec1, self.hdp)
        self.log(sim)
        self.assertAlmostEqual(sim, 0.0, delta=0.0002)
        sim = hellinger_distance(vec2, vec2, self.hdp)
        self.log(sim)
        self.assertAlmostEqual(sim, 0.0, delta=0.0002)
        sim = hellinger_distance(vec3, vec3, self.hdp)
        self.log(sim)
        self.assertAlmostEqual(sim, 0.0, delta=0.0002)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
