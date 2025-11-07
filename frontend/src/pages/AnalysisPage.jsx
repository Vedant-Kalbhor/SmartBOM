import React, { useState, useEffect } from 'react';
import { Card, Form, Input, Select, Button, Row, Col, message, Slider, Table } from 'antd';
import { PlayCircleOutlined } from '@ant-design/icons';
import { getWeldmentFiles, getBOMFiles, analyzeDimensionalClustering, analyzeBOMSimilarity } from '../services/api';
import { useNavigate } from 'react-router-dom';

const { Option } = Select;

const AnalysisPage = () => {
  const [form] = Form.useForm();
  const [bomForm] = Form.useForm();
  const [weldmentFiles, setWeldmentFiles] = useState([]);
  const [bomFiles, setBomFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    loadFiles();
  }, []);

  const loadFiles = async () => {
    try {
      const [weldmentResponse, bomResponse] = await Promise.all([
        getWeldmentFiles(),
        getBOMFiles()
      ]);
      setWeldmentFiles(weldmentResponse.data || []);
      setBomFiles(bomResponse.data || []);
    } catch (error) {
      console.error('Failed to load files:', error);
      message.error('Failed to load files');
    }
  };

  const onDimensionalAnalysis = async (values) => {
    try {
      setLoading(true);
      console.log('Starting dimensional clustering with values:', values);
      
      const response = await analyzeDimensionalClustering(values);
      console.log('Dimensional clustering response:', response.data);
      
      setAnalysisResults(response.data);
      message.success('Dimensional analysis completed successfully');
      
      // Navigate to results page with the complete analysis data
      navigate(`/results/${response.data.analysis_id}`, { 
        state: { 
          analysisResults: {
            clustering: response.data.clustering_result,
            bom_analysis: response.data.bom_analysis_result
          }
        } 
      });
    } catch (error) {
      console.error('Dimensional analysis error:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Analysis failed';
      message.error(`Analysis failed: ${errorMessage}`);
    } finally {
      setLoading(false);
    }
  };

  const onBOMAnalysis = async (values) => {
    try {
      setLoading(true);
      console.log('Starting BOM analysis with values:', values);
      
      const response = await analyzeBOMSimilarity(values);
      console.log('BOM analysis response:', response.data);
      
      setAnalysisResults(response.data);
      message.success('BOM analysis completed successfully');
      
      // Navigate to results page with the complete analysis data
      navigate(`/results/${response.data.analysis_id}`, { 
        state: { 
          analysisResults: {
            clustering: response.data.clustering_result,
            bom_analysis: response.data.bom_analysis_result
          }
        } 
      });
    } catch (error) {
      console.error('BOM analysis error:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Analysis failed';
      message.error(`Analysis failed: ${errorMessage}`);
    } finally {
      setLoading(false);
    }
  };

  // Helper function to check if files are uploaded
  const hasWeldmentFiles = weldmentFiles.length > 0;
  const hasBomFiles = bomFiles.length > 0;

  return (
    <div>
      <h1>Analysis</h1>
      
      {!hasWeldmentFiles && !hasBomFiles && (
        <Card style={{ marginBottom: 20 }}>
          <div style={{ textAlign: 'center', padding: '20px' }}>
            <h3>No Files Uploaded</h3>
            <p>Please upload weldment and BOM files first to run analysis.</p>
            <Button type="primary" onClick={() => navigate('/upload')}>
              Go to Upload
            </Button>
          </div>
        </Card>
      )}

      <Row gutter={16}>
        <Col span={12}>
          <Card 
            title="Dimensional Clustering Analysis" 
            loading={loading}
            extra={
              <Button 
                type="primary" 
                icon={<PlayCircleOutlined />}
                onClick={() => form.submit()}
                disabled={!hasWeldmentFiles}
              >
                Run Analysis
              </Button>
            }
          >
            {!hasWeldmentFiles ? (
              <div style={{ textAlign: 'center', padding: '20px' }}>
                <p>No weldment files uploaded.</p>
                <Button type="primary" onClick={() => navigate('/upload')}>
                  Upload Weldment Files
                </Button>
              </div>
            ) : (
              <Form
                form={form}
                layout="vertical"
                onFinish={onDimensionalAnalysis}
                initialValues={{
                  clustering_method: "kmeans",
                  tolerance: 0.1
                }}
              >
                <Form.Item
                  name="weldment_file_id"
                  label="Weldment File"
                  rules={[{ required: true, message: 'Please select a weldment file' }]}
                >
                  <Select placeholder="Select weldment file">
                    {weldmentFiles.map(file => (
                      <Option key={file.file_id} value={file.file_id}>
                        {file.filename} ({file.record_count} records)
                      </Option>
                    ))}
                  </Select>
                </Form.Item>
                
                <Form.Item
                  name="clustering_method"
                  label="Clustering Method"
                >
                  <Select>
                    <Option value="kmeans">K-Means</Option>
                    <Option value="hierarchical">Hierarchical</Option>
                    <Option value="dbscan">DBSCAN</Option>
                  </Select>
                </Form.Item>
                
                <Form.Item
                  name="n_clusters"
                  label="Number of Clusters (optional)"
                  help="Leave empty for automatic cluster detection"
                >
                  <Input type="number" min={2} max={20} placeholder="Auto-detect if empty" />
                </Form.Item>
                
                <Form.Item
                  name="tolerance"
                  label="Tolerance"
                  help="Higher tolerance allows more variation within clusters"
                >
                  <Slider
                    min={0.01}
                    max={1}
                    step={0.01}
                    marks={{
                      0.01: '0.01',
                      0.5: '0.5',
                      1: '1'
                    }}
                  />
                </Form.Item>

                <Form.Item>
                  <Button 
                    type="primary" 
                    htmlType="submit" 
                    icon={<PlayCircleOutlined />}
                    block
                    size="large"
                  >
                    Run Dimensional Clustering
                  </Button>
                </Form.Item>
              </Form>
            )}
          </Card>
        </Col>
        
        <Col span={12}>
          <Card 
            title="BOM Similarity Analysis"
            loading={loading}
            extra={
              <Button 
                type="primary" 
                icon={<PlayCircleOutlined />}
                onClick={() => bomForm.submit()}
                disabled={!hasBomFiles}
              >
                Run Analysis
              </Button>
            }
          >
            {!hasBomFiles ? (
              <div style={{ textAlign: 'center', padding: '20px' }}>
                <p>No BOM files uploaded.</p>
                <Button type="primary" onClick={() => navigate('/upload')}>
                  Upload BOM Files
                </Button>
              </div>
            ) : (
              <Form 
                form={bomForm}
                layout="vertical" 
                onFinish={onBOMAnalysis}
                initialValues={{
                  similarity_method: "jaccard",
                  threshold: 0.8
                }}
              >
                <Form.Item
                  name="bom_file_id"
                  label="BOM File"
                  rules={[{ required: true, message: 'Please select a BOM file' }]}
                >
                  <Select placeholder="Select BOM file">
                    {bomFiles.map(file => (
                      <Option key={file.file_id} value={file.file_id}>
                        {file.filename} ({file.record_count} records)
                      </Option>
                    ))}
                  </Select>
                </Form.Item>
                
                <Form.Item
                  name="similarity_method"
                  label="Similarity Method"
                >
                  <Select>
                    <Option value="jaccard">Jaccard Similarity</Option>
                    <Option value="cosine">Cosine Similarity</Option>
                    <Option value="weighted">Weighted Similarity</Option>
                  </Select>
                </Form.Item>
                
                <Form.Item
                  name="threshold"
                  label="Similarity Threshold"
                  help="Higher threshold shows only very similar BOMs"
                >
                  <Slider
                    min={0.1}
                    max={1}
                    step={0.1}
                    marks={{
                      0.1: '0.1',
                      0.5: '0.5',
                      1: '1'
                    }}
                  />
                </Form.Item>

                <Form.Item>
                  <Button 
                    type="primary" 
                    htmlType="submit" 
                    icon={<PlayCircleOutlined />}
                    block
                    size="large"
                  >
                    Run BOM Similarity Analysis
                  </Button>
                </Form.Item>
              </Form>
            )}
          </Card>
        </Col>
      </Row>

      {/* Quick Actions Card */}
      <Card title="Quick Actions" style={{ marginTop: 20 }}>
        <Row gutter={16}>
          <Col span={8}>
            <Card
              size="small"
              title="Upload Files"
              actions={[
                <Button 
                  type="link" 
                  onClick={() => navigate('/upload')}
                >
                  Go to Upload
                </Button>
              ]}
            >
              Upload weldment dimensions and BOM files for analysis
            </Card>
          </Col>
          <Col span={8}>
            <Card
              size="small"
              title="View Results"
              actions={[
                <Button 
                  type="link" 
                  onClick={() => navigate('/results')}
                >
                  View Results
                </Button>
              ]}
            >
              Explore previous analysis results and optimization suggestions
            </Card>
          </Col>
          <Col span={8}>
            <Card
              size="small"
              title="Dashboard"
              actions={[
                <Button 
                  type="link" 
                  onClick={() => navigate('/')}
                >
                  Go to Dashboard
                </Button>
              ]}
            >
              View overall statistics and project status
            </Card>
          </Col>
        </Row>
      </Card>

      {/* Analysis Tips */}
      <Card title="Analysis Tips" style={{ marginTop: 20 }}>
        <Row gutter={16}>
          <Col span={12}>
            <h4>Dimensional Clustering:</h4>
            <ul>
              <li>Use K-Means for well-separated, spherical clusters</li>
              <li>Use Hierarchical for nested cluster relationships</li>
              <li>Use DBSCAN for clusters with irregular shapes and noise</li>
              <li>Higher tolerance allows more dimensional variation within clusters</li>
            </ul>
          </Col>
          <Col span={12}>
            <h4>BOM Similarity:</h4>
            <ul>
              <li>Jaccard similarity compares component sets</li>
              <li>Cosine similarity considers component quantities</li>
              <li>Higher threshold shows only very similar BOMs</li>
              <li>Look for BOM pairs with similarity above 80% for consolidation</li>
            </ul>
          </Col>
        </Row>
      </Card>

      {/* Debug: Show raw results if available */}
      {analysisResults && process.env.NODE_ENV === 'development' && (
        <Card title="Raw Analysis Results (Debug)" style={{ marginTop: 20 }}>
          <details>
            <summary>Click to view raw API response</summary>
            <pre style={{ fontSize: '10px', maxHeight: '300px', overflow: 'auto' }}>
              {JSON.stringify(analysisResults, null, 2)}
            </pre>
          </details>
        </Card>
      )}
    </div>
  );
};

export default AnalysisPage;