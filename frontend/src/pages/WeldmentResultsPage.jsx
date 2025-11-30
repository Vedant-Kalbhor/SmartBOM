import React, { useState, useEffect } from 'react';
import {
  Card,
  Table,
  Tag,
  Progress,
  Alert,
  Button,
  Spin,
  message,
  Statistic,
  Row,
  Col
} from 'antd';
import { DownloadOutlined } from '@ant-design/icons';
import { saveAs } from 'file-saver';
import { getAnalysisResults } from '../services/api';
import { useParams, useLocation, useNavigate } from 'react-router-dom';

const WeldmentResultsPage = () => {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const { analysisId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();

  const hasCostSavings =
    !!results &&
    !!results.cost_savings &&
    results.cost_savings.has_cost_data &&
    Array.isArray(results.cost_savings.rows) &&
    results.cost_savings.rows.length > 0;

  useEffect(() => {
    if (location.state?.analysisResults?.weldment_pairwise) {
      setResults(location.state.analysisResults.weldment_pairwise);
      setLoading(false);
    } else if (analysisId) {
      loadAnalysisResults();
    } else {
      setLoading(false);
    }
    // eslint-disable-next-line
  }, [analysisId, location.state]);

  const loadAnalysisResults = async () => {
    try {
      setLoading(true);
      const response = await getAnalysisResults(analysisId);
      const data = response.data;
      const raw = data.raw || data;
      const weld =
        raw?.weldment_pairwise ||
        raw?.weldment_pairwise_result ||
        data.weldment_pairwise_result;
      setResults(weld || null);
    } catch (err) {
      console.error('Error loading weldment results:', err);
      message.error('Failed to load results');
    } finally {
      setLoading(false);
    }
  };

  const handleExportCSV = () => {
    try {
      const rows = results?.pairwise_table || [];
      if (!rows.length) {
        message.warning('No results to export');
        return;
      }

      if (hasCostSavings) {
        const savingsRows = results.cost_savings.rows || [];
        const header = [
          'Assembly A',
          'Assembly B',
          'Match %',
          'Matching Letters',
          'Matching Columns',
          'Unmatching Letters',
          'Cost A',
          'EAU A',
          'Cost B',
          'EAU B',
          'Old Price',
          'New Price',
          'Old-New Price',
          'Effective EAU',
          'Recommended Assembly',
          'Recommended Cost',
          'Total Cost Before',
          'Total Cost After',
          'Cost Savings',
          'Savings %'
        ];

        const keyFn = (a, b) => `${a}__${b}`;
        const costMap = {};
        savingsRows.forEach(r => {
          const k1 = keyFn(r.bom_a, r.bom_b);
          const k2 = keyFn(r.bom_b, r.bom_a);
          costMap[k1] = r;
          costMap[k2] = r;
        });

        const csvRows = [
          header.join(','),
          ...rows.map(r => {
            const pairKey = keyFn(r.bom_a, r.bom_b);
            const c = costMap[pairKey] || {};
            return [
              `"${(r.bom_a || '').replace(/"/g, '""')}"`,
              `"${(r.bom_b || '').replace(/"/g, '""')}"`,
              `${r.match_percentage ?? 0}`,
              `"${(r.matching_columns_letters || '').replace(/"/g, '""')}"`,
              `"${(r.matching_columns || []).join('; ').replace(/"/g, '""')}"`,
              `"${(r.unmatching_columns_letters || '').replace(/"/g, '""')}"`,
              c.cost_a ?? '',
              c.eau_a ?? '',
              c.cost_b ?? '',
              c.eau_b ?? '',
              c.old_price ?? '',
              c.new_price ?? '',
              c.old_new_price ?? '',
              c.effective_eau ?? '',
              c.recommended_assembly
                ? `"${String(c.recommended_assembly).replace(/"/g, '""')}"`
                : '',
              c.recommended_cost ?? '',
              c.total_cost_before ?? '',
              c.total_cost_after ?? '',
              c.cost_savings ?? '',
              c.savings_percent ?? ''
            ].join(',');
          })
        ].join('\n');

        const blob = new Blob([csvRows], { type: 'text/csv;charset=utf-8;' });
        saveAs(blob, `weldment-pairwise-cost-${analysisId || 'latest'}.csv`);
        message.success('Exported CSV with cost savings');
        return;
      }

      // Legacy export (no cost data)
      const header = [
        'Assembly A',
        'Assembly B',
        'Match %',
        'Matching Letters',
        'Matching Columns',
        'Unmatching Letters'
      ];
      const csvRows = [
        header.join(','),
        ...rows.map(r =>
          [
            `"${(r.bom_a || '').replace(/"/g, '""')}"`,
            `"${(r.bom_b || '').replace(/"/g, '""')}"`,
            `${r.match_percentage ?? 0}`,
            `"${(r.matching_columns_letters || '').replace(/"/g, '""')}"`,
            `"${(r.matching_columns || []).join('; ').replace(/"/g, '""')}"`,
            `"${(r.unmatching_columns_letters || '').replace(/"/g, '""')}"`,
          ].join(',')
        )
      ].join('\n');

      const blob = new Blob([csvRows], { type: 'text/csv;charset=utf-8;' });
      saveAs(blob, `weldment-pairwise-${analysisId || 'latest'}.csv`);
      message.success('Exported CSV');
    } catch (err) {
      console.error(err);
      message.error('Export failed');
    }
  };

  const columns = [
    {
      title: 'Assembly A',
      dataIndex: 'bom_a',
      key: 'bom_a',
      render: t => <span style={{ fontFamily: 'monospace' }}>{t}</span>
    },
    {
      title: 'Assembly B',
      dataIndex: 'bom_b',
      key: 'bom_b',
      render: t => <span style={{ fontFamily: 'monospace' }}>{t}</span>
    },
    {
      title: 'Match %',
      dataIndex: 'match_percentage',
      key: 'match_percentage',
      render: val => {
        const pct = Number(val) || 0;
        return (
          <Progress
            percent={Math.round(pct)}
            size="small"
            format={p => `${p.toFixed(1)}%`}
          />
        );
      }
    },
    {
      title: 'Matching (columns)',
      dataIndex: 'matching_columns',
      key: 'matching_columns',
      render: arr =>
        Array.isArray(arr)
          ? arr.map((c, i) => <Tag key={i}>{c}</Tag>)
          : null
    }
  ];

  const costColumns = hasCostSavings
    ? [
        {
          title: 'Assembly A',
          dataIndex: 'bom_a',
          key: 'bom_a',
          render: t => <span style={{ fontFamily: 'monospace' }}>{t}</span>
        },
        {
          title: 'Assembly B',
          dataIndex: 'bom_b',
          key: 'bom_b',
          render: t => <span style={{ fontFamily: 'monospace' }}>{t}</span>
        },
        {
          title: 'Match %',
          dataIndex: 'match_percentage',
          key: 'match_percentage',
          render: val => {
            const pct = Number(val) || 0;
            return (
              <span style={{ color: '#52c41a' }}>
                {pct.toFixed(1)}%
              </span>
            );
          }
        },
        {
          title: 'Cost A',
          dataIndex: 'cost_a',
          key: 'cost_a',
          render: v => (v != null ? Number(v).toLocaleString() : '-')
        },
        {
          title: 'EAU A',
          dataIndex: 'eau_a',
          key: 'eau_a',
          render: v => (v != null ? Number(v).toLocaleString() : '-')
        },
        {
          title: 'Cost B',
          dataIndex: 'cost_b',
          key: 'cost_b',
          render: v => (v != null ? Number(v).toLocaleString() : '-')
        },
        {
          title: 'EAU B',
          dataIndex: 'eau_b',
          key: 'eau_b',
          render: v => (v != null ? Number(v).toLocaleString() : '-')
        },
        {
          title: 'Old-New Price',
          dataIndex: 'old_new_price',
          key: 'old_new_price',
          render: v =>
            v != null
              ? Number(v).toLocaleString(undefined, {
                  maximumFractionDigits: 2
                })
              : '-'
        },
        {
          title: 'EAU (Replaced)',
          dataIndex: 'effective_eau',
          key: 'effective_eau',
          render: v => (v != null ? Number(v).toLocaleString() : '-')
        },
        {
          title: 'Recommended Assembly',
          dataIndex: 'recommended_assembly',
          key: 'recommended_assembly',
          render: t => <span style={{ fontFamily: 'monospace' }}>{t}</span>
        },
        {
          title: 'Total Cost Before',
          dataIndex: 'total_cost_before',
          key: 'total_cost_before',
          render: v =>
            v != null
              ? Number(v).toLocaleString(undefined, {
                  maximumFractionDigits: 2
                })
              : '-'
        },
        {
          title: 'Total Cost After',
          dataIndex: 'total_cost_after',
          key: 'total_cost_after',
          render: v =>
            v != null
              ? Number(v).toLocaleString(undefined, {
                  maximumFractionDigits: 2
                })
              : '-'
        },
        {
          title: 'Cost Savings',
          dataIndex: 'cost_savings',
          key: 'cost_savings',
          render: v =>
            v != null
              ? Number(v).toLocaleString(undefined, {
                  maximumFractionDigits: 2
                })
              : '-'
        },
        {
          title: 'Savings %',
          dataIndex: 'savings_percent',
          key: 'savings_percent',
          render: v =>
            v != null ? `${Number(v).toFixed(2)}%` : '-'
        }
      ]
    : [];

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 50 }}>
        <Spin size="large" />
        <p>Loading weldment comparison results...</p>
      </div>
    );
  }

  if (!results) {
    return (
      <div style={{ textAlign: 'center', padding: 40 }}>
        <Alert
          message="No Weldment Pairwise Results"
          description="Run a weldment one-to-one comparison to see results here."
          type="info"
          showIcon
          action={
            <Button type="primary" onClick={() => navigate('/analysis')}>
              Run Analysis
            </Button>
          }
        />
      </div>
    );
  }

  if (hasCostSavings) {
    const statsBlock = results.cost_savings.statistics || {};
    const totalPairs = results.pairwise_table?.length || 0;
    const totalPerfect = statsBlock.pair_count_100 || 0;
    const totalSavings = statsBlock.total_cost_savings || 0;
    const avgSavingsPercent = statsBlock.avg_savings_percent || 0;

    return (
      <div>
        <h1>Weldment One-to-One Comparison (with Cost & EAU)</h1>

        <Card style={{ marginBottom: 20 }}>
          <Row gutter={16} align="middle">
            <Col xs={24} sm={12} md={6}>
              <Statistic title="Pairs Above Threshold" value={totalPairs} />
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Statistic title="100% Matching Pairs" value={totalPerfect} />
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Statistic
                title="Total Savings (All Replacements)"
                value={totalSavings}
                precision={2}
              />
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Statistic
                title="Avg Savings %"
                value={avgSavingsPercent}
                precision={2}
                suffix="%"
              />
            </Col>
          </Row>

          <div style={{ marginTop: 16, textAlign: 'right' }}>
            <Button icon={<DownloadOutlined />} onClick={handleExportCSV}>
              Export CSV (with cost)
            </Button>
          </div>
        </Card>

        <Card title="Pairwise Dimension Comparison" style={{ marginBottom: 20 }}>
          <Table
            columns={columns}
            dataSource={results.pairwise_table || []}
            pagination={false}
            rowKey={(r, i) => `${r.bom_a}-${r.bom_b}-${i}`}
          />
        </Card>

        <Card title="Cost & EAU Savings for 100% Matches">
          <Table
            columns={costColumns}
            dataSource={results.cost_savings.rows || []}
            pagination={false}
            rowKey={(r, i) => `cost-${r.bom_a}-${r.bom_b}-${i}`}
          />
        </Card>
      </div>
    );
  }

  // Legacy layout (no cost/EAU columns)
  return (
    <div>
      <h1>Weldment One-to-One Comparison</h1>

      <Card style={{ marginBottom: 20 }}>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}
        >
          <div>
            <div
              style={{
                fontSize: 22,
                fontWeight: 'bold',
                color: '#1890ff'
              }}
            >
              {results.pairwise_table?.length || 0} pairs
            </div>
            <div style={{ color: '#666' }}>Pairs above threshold</div>
          </div>

          <Button icon={<DownloadOutlined />} onClick={handleExportCSV}>
            Export CSV
          </Button>
        </div>
      </Card>

      <Card title="Pairwise Table">
        <Table
          columns={columns}
          dataSource={results.pairwise_table || []}
          pagination={false}
          rowKey={(r, i) => `${r.bom_a}-${r.bom_b}-${i}`}
        />
      </Card>
    </div>
  );
};

export default WeldmentResultsPage;
