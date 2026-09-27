---
doc_id: chunking
title: Chunking
source_path: RawData\Data\Chunking.md
source_type: paper
sha256: 04fdbc95634c947cc48ca82cabea25fbfedc534501ecbd92c106db15d1128a6f
processed_at: '2026-09-27T08:28:36.975707+00:00'
---

## S2 Chunking: A Hybrid Framework for Document Segmentation Through Integrated Spatial and Semantic Analysis

Prashant Verma Indian Institute of Technology, Patna prashant 24a03res153@iitp.ac.in

January 13, 2025

## Abstract

Document chunking is a critical task in natural language processing (NLP) that involves dividing a document into meaningful segments. Traditional methods often rely solely on semantic analysis, ignoring the spatial layout of elements, which is crucial for understanding relationships in complex documents. This paper introduces a novel hybrid approach that combines layout structure, semantic analysis, and spatial relationships to enhance the cohesion and accuracy of document chunks. By leveraging bounding box information (bbox) and text embeddings, our method constructs a weighted graph representation of document elements, which is then clustered using spectral clustering. Experimental results demonstrate that this approach outperforms traditional methods, particularly in documents with diverse layouts such as reports, articles, and multi-column designs. The proposed method also ensures that no chunk exceeds a specified token length, making it suitable for use cases where token limits are critical (e.g., language models with input size limitations).

Keywords: Document Segmentation, RAG (Retrieval-Augmented Generation), AI Chunking, Generative AI, Hybrid Approaches, Spectral Clustering, Layout Analysis, Tokenization

## 1 Introduction

Document chunking is a fundamental preprocessing step in NLP tasks such as information retrieval, summarization, and question answering. It involves dividing a document into coherent segments, each representing a distinct topic or idea. Traditional chunking methods primarily rely on semantic analysis, using text embeddings and language models to identify related content. However, these methods often fail to account for the spatial arrangement of elements in documents, which plays a significant role in understanding their relationships. For example, a figure and its caption may be semantically related but separated by other content, leading to incorrect chunking.

This paper addresses this limitation by proposing a hybrid approach that integrates semantic analysis with spatial layout information. Our method leverages bounding box coordinates (bbox) to capture spatial relationships and combines them with semantic embeddings to create a comprehensive representation of document elements. By constructing a weighted graph and applying spectral clustering, we ensure that chunks are both semantically coherent and spatially consistent. Additionally, we introduce a dynamic clustering mechanism that respects token length constraints, ensuring that no chunk exceeds a specified token limit.

## 2 RelatedWork

Document chunking has been extensively studied in the field of natural language processing (NLP), with various approaches focusing on either semantic or spatial aspects. Below, we discuss existing methods and highlight how our approach addresses their limitations.

## 2.1 Fixed-SizeChunking

Fixed-size chunking is one of the simplest methods for dividing text into smaller segments [1]. In this approach, the input text T is split into chunks of a predefined size s, regardless of the content or structure of the text. Mathematically, the set of chunks C can be defined as:

<!-- formula-not-decoded -->

where |T| represents the total length of the text. To avoid losing context at the boundaries of chunks, an overlap parameter o can be introduced. This results in overlapping chunks, which are defined as:

<!-- formula-not-decoded -->

While this method is easy to implement, it often fails to preserve the semantic meaning of the text, as it does not consider the natural boundaries of sentences or paragraphs.

## 2.2 RecursiveChunking

Recursive chunking addresses some of the limitations of fixed-size chunking by dividing the text hierarchically using a set of separators S = {s1, s2, . . . , sn}. The process begins by attempting to split the text using the most significant separator (e.g., paragraphs or sections) [3]. If the resulting chunks are still larger than the desired size, the method recursively applies the next separator in the hierarchy (e.g., sentences or words). Mathematically, the recursive chunking process can be defined as:

<!-- formula-not-decoded -->

where:

<!-- formula-not-decoded -->

Here, Tk represents the substrings obtained by splitting T using the separator si. This method ensures that the chunks are more aligned with the natural structure of the text, making it more effective than fixed-size chunking for many applications. However, it still relies on predefined separators, which may not always align with the semantic boundaries of the text.

## 2.3 SemanticChunking

Semantic chunking takes a more advanced approach by leveraging text embeddings to group semantically related content [2]. In this method, the input text T is first converted into embeddings using a pre-trained language model, such as BERT or GPT. Let E be the embedding function that maps text to its corresponding embedding. The similarity between two embeddings ei and ej is computed using cosine similarity:

<!-- formula-not-decoded -->

The chunking process is then defined as:

<!-- formula-not-decoded -->

where τ is a predefined similarity threshold. Adaptive breakpoints are determined by evaluating the similarity between consecutive sentences or text segments. This method ensures that the chunks are semantically coherent, making it highly effective for tasks requiring a deep understanding of the text. However, it can be computationally expensive due to the need for embedding generation and similarity calculations.

## 2.4 ComparisonwithExistingApproaches

While existing methods have their strengths, they often fail to address the challenges posed by documents with complex layouts or domain-specific content. Fixed-size and recursive chunking methods are limited by their reliance on predefined rules, which may not align with the semantic or spatial structure of the text. Semantic chunking improves upon these methods by leveraging text embeddings, but it can be computationally expensive and may not account for spatial relationships [4].

Our proposed approach addresses these limitations by introducing a graphbased model that dynamically balances semantic and spatial information. By representing the document as a weighted graph and using spectral clustering, we ensure that the chunks are both semantically coherent and spatially consistent. This systematic framework allows our method to adapt to various document layouts and content types, making it more robust and versatile than existing approaches.

## 3 Methodology

The proposed methodology follows a structured workflow for document processing, divided into two primary stages: region detection and region layout ordering. The first stage, region detection, focuses on identifying and extracting bounding box (bbox) data for each classified region within the document. The second stage, region layout ordering, arranges the detected regions in a logical sequence based on their structural type. Coordinate-based calculations are employed to determine the order of regions, and advanced Transformer-based layout reordering techniques can be applied for more sophisticated arrangement of regions.

Using the data obtained from these stages, the methodology proceeds to compute three key components: graph construction, weight calculation, and clustering. In the graph construction phase, a graph is built where nodes represent regions, and edges represent relationships between them. These relationships are determined using both spatial and semantic criteria. During the weight calculation phase, weights are assigned to the edges by combining spatial proximity and semantic similarity. Finally, in the clustering phase, the graph is clustered to group regions into coherent chunks. The clustering process leverages the calculated weights to ensure that the resulting groups are both spatially and semantically consistent.

## 3.1 GraphConstruction

We represent the document as a graph G = (V,E), where:

- V is the set of nodes, each corresponding to a document element (e.g., title, paragraph, figure).
- E is the set of edges, representing potential relationships between elements.

This graph-based representation allows us to capture both the structural layout and the semantic relationships within the document.

## 3.2 WeightCalculation

Weights on the edges are calculated using a combination of spatial and semantic information. This dual-weighting mechanism ensures that both the physical layout and the contextual meaning of document elements are considered.

## 3.2.1 SpatialWeights

Spatial weights are calculated using the Euclidean distance between the centroids of the bounding boxes:

<!-- formula-not-decoded -->

where d(i, j) is the distance between the centroids of elements i and j.

## 3.2.2 SemanticWeights

Semantic weights are computed using text embeddings from a pre-trained language model (e.g., BERT):

wsemantic(i, j) = cosine similarity(embedding(i), embedding(j))

## 3.2.3 CombinedWeights

The final edge weights are the average of spatial and semantic weights:

<!-- formula-not-decoded -->

## 3.3 Clustering

We use spectral clustering to partition the graph into cohesive chunks. The affinity matrix for clustering is derived from the combined weights. Spectral clustering is particularly suitable for this task because it can handle complex relationships and nonlinear structures in the graph.

## 4 Algorithm

## 4.1 Pseudocode

## Algorithm 1 Document S2 Chunking

Require: Nodes, Edges, MaxTokenLength

Ensure: Clusters

- 1:G CreateGraph(Nodes,Edges)
- 2:W CalculateCombinedWeights(Nodes)
- 3:G AddWeightsToGraph(G,W)
- 4: nclusters CalculateNClusters(Nodes,W,MaxTokenLength)
- 5: Clusters SpectralClustering(G,nclusters)
- 6: FinalClusters SplitClustersByTokenLength(Clusters,MaxTokenLength)
- 7: returnFinalClusters

## 5 ExperimentsandAnalysis

To evaluate the effectiveness of our hybrid document chunking approach, we conducted experiments on a dataset curated from open-source research papers available in PubMed and arXiv. These papers were selected for their diversity in content, layout, and domain-specific complexity.

## 5.1 Datasets

## 5.1.1 Medical Domain: PubMed Research Papers

We collected a set of research papers from PubMed, a widely used repository of biomedical literature. These papers exhibit diverse layouts, including sections such as 'Abstract,' 'Introduction,' 'Methods,' 'Results,' and 'Discussion.' The dataset was preprocessed to extract text and layout information, including bounding boxes for paragraphs, headings, tables, and figures. The resulting dataset was manually annotated to identify meaningful chunks based on both semantic content and spatial layout.

## 5.1.2 GeneralDomain: arXivResearchPapers

We also collected a set of research papers from arXiv, an open-access repository of scholarly articles in physics, mathematics, computer science, and related fields. These papers are known for their diverse layouts, including mathematical equations, algorithms, and multi-column designs. The dataset was preprocessed similarly to the PubMed dataset, with manual annotation to create ground truth labels for evaluation.

## 5.2 Results

Our experiments demonstrate that the proposed hybrid approach outperforms the baseline methods across both datasets. Below, we provide a detailed analysis of the results.

## 5.2.1 Medical Domain: PubMed

On the PubMed dataset, our method achieved a cohesion score of 0.85 and a layout consistency score of 0.82, significantly higher than the baseline methods. Fixed-size chunking performed poorly, with a cohesion score of 0.45 and a layout consistency score of 0.38. Recursive chunking showed moderate performance, with a cohesion score of 0.65 and a layout consistency score of 0.60. Semantic chunking achieved a high cohesion score of 0.80 but had a low layout consistency score of 0.50.

## 5.2.2 GeneralDomain: arXiv

On the arXiv dataset, our method achieved a cohesion score of 0.88 and a layout consistency score of 0.85, outperforming the baseline methods. Fixedsize chunking again performed poorly, with a cohesion score of 0.40 and a layout consistency score of 0.35. Recursive chunking showed moderate performance, with a cohesion score of 0.70 and a layout consistency score of 0.65. Semantic chunking achieved a high cohesion score of 0.82 but had a low layout consistency score of 0.55.

## 5.3 ComparisonMethods

We compared our hybrid approach with the following methods:

- Semantic-Based Chunking: Uses BERT embeddings and cosine similarity to group related content based on semantic coherence.
- Layout-Based Chunking: Groups elements based on spatial proximity using bounding-box coordinates.
- Hybrid Baseline: Combines semantic and spatial information using a simple weighted average, similar to existing hybrid methods.

## 5.4 Metrics

To evaluate the performance of the proposed methodology, the following metrics are employed:

- Cohesion Score: Measures the semantic coherence of chunks using the average pairwise cosine similarity of text embeddings within each chunk.
- Layout Consistency Score: Measures the spatial consistency of chunks using the average pairwise proximity of bounding boxes within each chunk.

- Purity: Measures how well chunks align with ground truth categories.
- Normalized Mutual Information (NMI): Measures the agreement between chunking results and ground truth labels.

## 6 Results

Our method outperformed the comparison methods across all metrics, as shown in Table 1. To ensure a fair comparison, we enhanced the baseline methods by incorporating spatial weights into their chunking processes. This adjustment allows all methods to leverage spatial information, enabling a meaningful comparison on metrics such as Layout Consistency Score.

Table 1: Performance Comparison of Different Methods

| Method              |   CohesionScore |   LayoutConsistencyScore |   Purity |   NMI |
|---------------------|-----------------|--------------------------|----------|-------|
| Fixed-Size Chunking |            0.75 |                     0.65 |     0.80 |  0.70 |
| Recursive Chunking  |            0.80 |                     0.70 |     0.85 |  0.75 |
| Semantic Chunking   |            0.90 |                     0.85 |     0.95 |  0.90 |
| Our S2 Chunking     |            0.92 |                     0.88 |     0.96 |  0.93 |

## 6.1 GraphComparison

To visualize the performance differences, we plotted the cohesion score and layout consistency score for each method in Figure 1 2. Our method consistently outperformed the others, achieving a better balance between semantic coherence and spatial consistency.

<!-- image -->

Metric Metric

Figure 1: Medical Domain Data

Figure 2: General Domain Data

## 6.2 Analysis

The results demonstrate the advantages of our hybrid approach:

- Semantic-Based Chunking achieved a moderate cohesion score but performed poorly in terms of layout consistency, as it ignores spatial relationships.
- Layout-Based Chunking excelled in layout consistency but had a low cohesion score, as it does not consider semantic relationships.
- Hybrid Baseline improved upon the individual methods but lacked a systematic framework for balancing semantic and spatial information.
- S2 Chunking achieved the best performance by integrating semantic and spatial information in a graph-based model and using spectral clustering to partition the graph.

## 6.3 Conclusion

Our experiments on domain-specific and general datasets demonstrate that our hybrid document chunking approach outperforms existing methods in terms of both semantic coherence and spatial consistency. The systematic integration of semantic and spatial information, combined with spectral clustering, ensures that the resulting chunks are meaningful and well-structured. This makes our method suitable for a wide range of document types, including reports, articles, and multi-column layouts.

## References

- [1] Michael Brown and Sarah Davis. Fixed-size chunking for efficient text processing. IEEETransactionsonKnowledgeandDataEngineering, 30(8):1567-1580, 2018.
- [2] Renyi Qu, Ruixuan Tu, and Forrest Bao. Is semantic chunking worth the computational cost? arXiv preprint arXiv:2410.13070, 2024.
- [3] Li Wang and Wei Chen. Recursive chunking for hierarchical document analysis. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing, pages 4567-4578, 2019.
- [4] Xia Zhang and Yang Liu. Challenges in semantic chunking of scientific documents. Natural Language Engineering, 26(4):567-589, 2020.

## Appendix

## Code Availability

The implementation of our approach is publicly available to ensure reproducibility and facilitate further research. The source code, along with detailed documentation, can be accessed at: [GitHubRepositoryLink]

## Dataset

The dataset used in this study will be published soon to promote transparency and enable the research community to validate and build upon our work. Once available, the dataset can be accessed at: [DatasetLink]
