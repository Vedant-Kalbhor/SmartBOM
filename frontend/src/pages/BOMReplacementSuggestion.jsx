import React, { useEffect, useState } from 'react';
import { Card, Table, Spin, Alert, Button, message } from 'antd';
import { ArrowLeftOutlined, DownloadOutlined } from '@ant-design/icons';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { getAnalysisResults } from '../services/api';
import { saveAs } from 'file-saver';

const BOMReplacementSuggestion = () => {
  const { analysisId, bomA, bomB } = useParams();
  const navigate = useNavigate();
  const location = useLocation();

  const [loading, setLoading] = useState(true);
  const [rows, setRows] = useState([]);

  useEffect(() => {
    // Try to use data passed from BOMResultsPage (state)
    const stateBomAnalysis =
      location.state?.analysisResults?.bom_analysis ||
      location.state?.analysisResults?.bom_analysis_result ||
      location.state?.bom_analysis ||
      location.state?.bom_analysis_result;

    if (stateBomAnalysis?.component_replacement_table) {
      filterRows(stateBomAnalysis.component_replacement_table);
      setLoading(false);
    } else {
      loadFromApi();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysisId, bomA, bomB, location.state]);

  const loadFromApi = async () => {
    try {
      setLoading(true);
      const response = await getAnalysisResults(analysisId);
      const data = response.data;

      // Data is stored in Mongo under data.raw.bom_analysis
      const bomAnalysis =
        data.raw?.bom_analysis || // from Mongo document
        data.bom_analysis_result || // direct API result (when not using Mongo)
        data.bom_analysis ||
        data;

      const table = bomAnalysis?.component_replacement_table || [];
      filterRows(table);
    } catch (error) {
      console.error('Error loading replacement suggestions:', error);
      message.error('Failed to load replacement suggestions');
    } finally {
      setLoading(false);
    }
  };

  const filterRows = (table) => {
    const filtered = (table || []).filter(
      (r) =>
        String(r.bom_a) === String(bomA) &&
        String(r.bom_b) === String(bomB)
    );
    setRows(filtered);
  };

  const handleExportCsv = () => {
    if (!rows || rows.length === 0) {
      message.warning('No replacement suggestions to export');
      return;
    }

    const header = [
      'Replace_In_BOM',
      'Replace_Out',
      'Replace_In_With',
      'New_MatchPct',
      'DeltaPct',
      'NewMatchedCount',
      'NewTotalAfter',
      'Direction',
    ];

    const body = rows.map((r) => [
      r.Replace_In_BOM,
      r.Replace_Out,
      r.Replace_In_With,
      r.New_MatchPct,
      r.DeltaPct,
      r.NewMatchedCount,
      r.NewTotalAfter,
      r.Direction,
    ]);

    const csvContent = [header, ...body]
      .map((row) => row.join(','))
      .join('\n');

    const blob = new Blob([csvContent], {
      type: 'text/csv;charset=utf-8;',
    });
    saveAs(
      blob,
      `bom-replacement-${bomA}-${bomB}-${analysisId || 'latest'}.csv`
    );
  };

  const columns = [
    {
      title: 'Replace In BOM',
      dataIndex: 'Replace_In_BOM',
      key: 'Replace_In_BOM',
      width: 140,
    },
    {
      title: 'Replace Out',
      dataIndex: 'Replace_Out',
      key: 'Replace_Out',
      width: 160,
    },
    {
      title: 'Replace In With',
      dataIndex: 'Replace_In_With',
      key: 'Replace_In_With',
      width: 160,
    },
    {
      title: 'New Match %',
      dataIndex: 'New_MatchPct',
      key: 'New_MatchPct',
      width: 130,
      render: (value) => `${value?.toFixed ? value.toFixed(2) : value} %`,
    },
    {
      title: 'Δ Match %',
      dataIndex: 'DeltaPct',
      key: 'DeltaPct',
      width: 120,
      render: (value) => `${value?.toFixed ? value.toFixed(2) : value} %`,
    },
    {
      title: 'New Matched Count',
      dataIndex: 'NewMatchedCount',
      key: 'NewMatchedCount',
      width: 150,
    },
    {
      title: 'New Total After',
      dataIndex: 'NewTotalAfter',
      key: 'NewTotalAfter',
      width: 140,
    },
    {
      title: 'Direction',
      dataIndex: 'Direction',
      key: 'Direction',
      width: 120,
    },
  ];

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 50 }}>
        <Spin size="large" />
        <p>Loading BOM replacement suggestions.</p>
      </div>
    );
  }

  if (!rows || rows.length === 0) {
    return (
      <div>
        <Button
          type="link"
          icon={<ArrowLeftOutlined />}
          onClick={() => navigate(-1)}
          style={{ marginBottom: 16, paddingLeft: 0 }}
        >
          Back to BOM Results
        </Button>
        <Alert
          message="No Replacement Suggestions"
          description="No component-level replacement suggestions were generated for this BOM pair."
          type="info"
          showIcon
        />
      </div>
    );
  }

  return (
    <div>
      <Button
        type="link"
        icon={<ArrowLeftOutlined />}
        onClick={() => navigate(-1)}
        style={{ marginBottom: 16, paddingLeft: 0 }}
      >
        Back to BOM Results
      </Button>

      <h1>BOM Replacement Suggestions</h1>
      <p style={{ color: '#555', marginBottom: 16 }}>
        For BOM pair:&nbsp;
        <strong>{bomA}</strong> &nbsp;↔&nbsp; <strong>{bomB}</strong>
      </p>

      <Card
        style={{ marginBottom: 20 }}
        title="One-to-One Replacement Analysis"
        extra={
          <Button
            icon={<DownloadOutlined />}
            onClick={handleExportCsv}
            size="small"
          >
            Export CSV
          </Button>
        }
      >
        <Table
          columns={columns}
          dataSource={rows}
          pagination={{ pageSize: 20 }}
          rowKey={(record, index) =>
            `${record.Replace_In_BOM}-${record.Replace_Out}-${record.Replace_In_With}-${record.Direction}-${index}`
          }
          scroll={{ x: 1000 }}
        />
      </Card>
    </div>
  );
};

export default BOMReplacementSuggestion;

