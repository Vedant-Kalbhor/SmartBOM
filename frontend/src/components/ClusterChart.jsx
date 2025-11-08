// import React from 'react';
// import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// const ClusterChart = ({ data, xKey, yKey }) => {
//   if (!data || data.length === 0) {
//     return <div>No data available for visualization</div>;
//   }

//   // Transform data for Recharts
//   const chartData = data.map((item, index) => ({
//     x: item[xKey],
//     y: item[yKey],
//     cluster: `Cluster ${index % 5}`,
//     name: item.assy_pn || `Item ${index}`
//   }));

//   const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#0088fe'];

//   return (
//     <ResponsiveContainer width="100%" height={400}>
//       <ScatterChart
//         data={chartData}
//         margin={{
//           top: 20,
//           right: 20,
//           bottom: 20,
//           left: 20,
//         }}
//       >
//         <CartesianGrid />
//         <XAxis 
//           type="number" 
//           dataKey="x" 
//           name={xKey}
//           label={{ value: xKey, position: 'insideBottom', offset: -5 }}
//         />
//         <YAxis 
//           type="number" 
//           dataKey="y" 
//           name={yKey}
//           label={{ value: yKey, angle: -90, position: 'insideLeft' }}
//         />
//         <ZAxis type="number" dataKey="z" range={[50, 500]} name="size" />
//         <Tooltip cursor={{ strokeDasharray: '3 3' }} />
//         <Legend />
//         <Scatter name="Weldments" data={chartData} fill="#8884d8" />
//       </ScatterChart>
//     </ResponsiveContainer>
//   );
// };

// export default ClusterChart;


// import React from 'react';
// import {
//   ScatterChart, Scatter, XAxis, YAxis,
//   CartesianGrid, Tooltip, Legend, ResponsiveContainer
// } from 'recharts';

// const ClusterChart = ({ data, xKey, yKey }) => {
//   if (!data || data.length === 0) {
//     return <div>No data available for visualization</div>;
//   }

//   // Group data by cluster number
//   const clusters = {};
//   console.log(data);
//   data.forEach((item, index) => {
//     const clusterId = item.cluster ?? 0; // handle undefined gracefully
//     if (!clusters[clusterId]) clusters[clusterId] = [];
//     clusters[clusterId].push({
//       x: item[xKey],
//       y: item[yKey],
//       name: item.assy_pn,
//     });
//   });

//   // Predefined colors — automatically cycles if there are more clusters
//   const colors = [
//     '#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#0088fe',
//     '#ff1493', '#00c49f', '#a83279', '#7fffd4', '#ff4500'
//   ];

//   return (
//     <ResponsiveContainer width="100%" height={400}>
//       <ScatterChart
//         margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
//       >
//         <CartesianGrid />
//         <XAxis
//           type="number"
//           dataKey="x"
//           name={xKey}
//           label={{ value: xKey, position: 'insideBottom', offset: -5 }}
//         />
//         <YAxis
//           type="number"
//           dataKey="y"
//           name={yKey}
//           label={{ value: yKey, angle: -90, position: 'insideLeft' }}
//         />
//         <Tooltip cursor={{ strokeDasharray: '3 3' }} />
//         <Legend />

//         {/* Render each cluster separately with its color */}
//         {Object.keys(clusters).map((clusterId, i) => (
//           <Scatter
//             key={clusterId}
//             name={`Cluster ${clusterId}`}
//             data={clusters[clusterId]}
//             fill={colors[i % colors.length]}
//           />
//         ))}
//       </ScatterChart>
//     </ResponsiveContainer>
//   );
// };

// export default ClusterChart;




import React from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

export default function ClusterChart({ data }) {
  if (!data || data.length === 0) {
    return (
      <div className="text-gray-500 text-center py-8">
        No cluster visualization data available.
      </div>
    );
  }

  // ✅ Auto-detect keys
  const sample = data[0];
  const xKey = sample.PC1 !== undefined ? "PC1" : Object.keys(sample)[2];
  const yKey = sample.PC2 !== undefined ? "PC2" : Object.keys(sample)[3];

  // ✅ Unique cluster IDs
  const clusters = [...new Set(data.map((d) => d.cluster))];

  const COLORS = ["#8884d8", "#82ca9d", "#ffc658", "#ff8042", "#0088FE", "#00C49F"];

  return (
    <ResponsiveContainer width="100%" height={400}>
      <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
        <CartesianGrid />
        <XAxis
          type="number"
          dataKey={xKey}
          name={xKey}
          label={{ value: xKey, position: "bottom" }}
        />
        <YAxis
          type="number"
          dataKey={yKey}
          name={yKey}
          label={{ value: yKey, angle: -90, position: "left" }}
        />
        <Tooltip
          cursor={{ strokeDasharray: "3 3" }}
          formatter={(value, name) => [value.toFixed(3), name]}
          contentStyle={{ backgroundColor: "#f9fafb", borderRadius: "8px" }}
          labelFormatter={(label) => `Part: ${label}`}
        />
        <Legend />
        {clusters.map((clusterId, i) => (
          <Scatter
            key={clusterId}
            name={`Cluster ${clusterId}`}
            data={data.filter((d) => d.cluster === clusterId)}
            fill={COLORS[i % COLORS.length]}
          />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  );
}
